import os
import json
from typing import Dict, Any
from dotenv import load_dotenv
load_dotenv()

try:
    from web3 import Web3
    WEB3_AVAILABLE = True
except Exception:
    WEB3_AVAILABLE = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
CONTRACTS_DIR = os.path.join(PROJECT_ROOT, "contracts")
os.makedirs(CONTRACTS_DIR, exist_ok=True)
FALLBACK_LEDGER = os.path.join(CONTRACTS_DIR, "ledger.json")

class BlockchainManager:
    def __init__(self):
        self.provider_url = os.getenv("WEB3_PROVIDER_URL", "http://127.0.0.1:8545")
        self.account_address = os.getenv("ACCOUNT_ADDRESS", "")
        self.private_key = os.getenv("PRIVATE_KEY", "")
        self.contract_address = os.getenv("CONTRACT_ADDRESS", "")
        self.contract_abi = None
        self.w3 = None
        self.contract = None

        if WEB3_AVAILABLE:
            self.w3 = Web3(Web3.HTTPProvider(self.provider_url))
            abi_path = os.path.join(CONTRACTS_DIR, "VerificationContract.json")
            if os.path.exists(abi_path):
                try:
                    with open(abi_path, "r", encoding="utf-8") as f:
                        contract_data = json.load(f)
                        self.contract_abi = contract_data.get("abi")
                except Exception:
                    self.contract_abi = None
            if self.contract_abi and self.contract_address:
                try:
                    self.contract = self.w3.eth.contract(address=self.contract_address, abi=self.contract_abi)
                except Exception:
                    self.contract = None

    def is_connected(self) -> bool:
        if self.w3:
            try:
                return self.w3.is_connected()
            except Exception:
                return False
        return True  # fallback: consider local ledger always available

    def store_verification_result(self, applicant_id: str, data_hash: str, risk_score: int, risk_category: str, probability_of_default: float) -> str:
        """
        Store verification result on-chain if possible; otherwise write to local JSON ledger.
        Returns tx hash string on success or an error string.
        """
        # Try on-chain
        if self.w3 and self.contract:
            if not (self.account_address and self.private_key):
                return "Error: ACCOUNT_ADDRESS or PRIVATE_KEY missing in .env"
            try:
                prob_int = int(max(0.0, min(1.0, probability_of_default)) * 10000)
                txn = self.contract.functions.storeVerificationResult(
                    applicant_id, data_hash, int(risk_score), risk_category, prob_int
                ).build_transaction({
                    "from": self.account_address,
                    "nonce": self.w3.eth.get_transaction_count(self.account_address),
                    "gas": 2000000,
                    "gasPrice": self.w3.eth.gas_price
                })
                signed = self.w3.eth.account.sign_transaction(txn, private_key=self.private_key)
                tx_hash = self.w3.eth.send_raw_transaction(signed.rawTransaction)
                receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
                return receipt.transactionHash.hex()
            except Exception as e:
                return f"Error: {str(e)}"

        # Fallback: local JSON ledger
        ledger = []
        if os.path.exists(FALLBACK_LEDGER):
            try:
                with open(FALLBACK_LEDGER, "r", encoding="utf-8") as f:
                    ledger = json.load(f)
            except Exception:
                ledger = []
        entry = {
            "applicant_id": applicant_id,
            "data_hash": data_hash,
            "risk_score": int(risk_score),
            "risk_category": risk_category,
            "probability_of_default": float(probability_of_default),
            "timestamp": __import__("datetime").datetime.utcnow().isoformat()
        }
        ledger.append(entry)
        try:
            with open(FALLBACK_LEDGER, "w", encoding="utf-8") as f:
                json.dump(ledger, f, indent=2)
            return "LOCAL_LEDGER_OK"
        except Exception as e:
            return f"Error: {str(e)}"

    def get_verification_result(self, applicant_id: str) -> Dict[str, Any]:
        """
        Read verification result from chain or local ledger.
        Returns dict or {'error': '...'}.
        """
        if self.w3 and self.contract:
            try:
                res = self.contract.functions.getVerificationResult(applicant_id).call()
                return {
                    "data_hash": res[0],
                    "risk_score": res[1],
                    "risk_category": res[2],
                    "probability_of_default": res[3] / 10000.0,
                    "timestamp": res[4]
                }
            except Exception as e:
                return {"error": str(e)}

        # Local ledger lookup
        if os.path.exists(FALLBACK_LEDGER):
            try:
                with open(FALLBACK_LEDGER, "r", encoding="utf-8") as f:
                    ledger = json.load(f)
                # return latest entry for applicant
                for entry in reversed(ledger):
                    if entry.get("applicant_id") == applicant_id:
                        return entry
            except Exception as e:
                return {"error": str(e)}
        return {"error": "Not found"}
