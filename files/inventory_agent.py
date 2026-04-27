from dataclasses import dataclass
from typing import Dict, List

@dataclass
class Part:
    name: str
    stock: int
    reorder_threshold: int
    pending_order: bool = False

class InventoryAgent:
    """
    Inventory Agent checks and reserves spare parts availability for repairs.
    """
    def __init__(self):
        # Initial simulated inventory
        self.inventory: Dict[str, Part] = {
            "Bearing": Part(name="Bearing", stock=15, reorder_threshold=5),
            "Thermal Paste": Part(name="Thermal Paste", stock=3, reorder_threshold=5),
            "Lubricant": Part(name="Lubricant", stock=20, reorder_threshold=10),
            "Valve": Part(name="Valve", stock=0, reorder_threshold=2), # Intentionally out of stock
            "Wiring Kit": Part(name="Wiring Kit", stock=5, reorder_threshold=2),
            "General Kit": Part(name="General Kit", stock=50, reorder_threshold=10)
        }

        # Map fault types to required parts
        self.fault_part_map = {
            "Bearing Wear": "Bearing",
            "Overheating": "Thermal Paste",
            "Lubrication Failure": "Lubricant",
            "Pressure Drop": "Valve",
            "Electrical Fault": "Wiring Kit",
            "Imbalance/Misalignment": "Bearing", # also uses bearings often
        }

    def check_and_reserve(self, fault_type: str) -> tuple[bool, str]:
        """
        Checks if the required part for a fault is in stock.
        If yes, decrements stock and returns (True, Part Name).
        If no, returns (False, Note).
        """
        part_name = self.fault_part_map.get(fault_type, "General Kit")
        
        if part_name not in self.inventory:
            return True, f"Specialized part ordered for {fault_type}"

        part = self.inventory[part_name]

        if part.stock > 0:
            part.stock -= 1
            
            # Check threshold
            if part.stock <= part.reorder_threshold and not part.pending_order:
                part.pending_order = True
                print(f"[InventoryAgent] Alert: {part.name} is low on stock ({part.stock}). Order placed.")
                
            return True, part_name
        else:
            if not part.pending_order:
                part.pending_order = True
                print(f"[InventoryAgent] Alert: {part.name} is OUT OF STOCK. Order placed.")
            return False, f"Waiting on parts: {part_name}"
    
    def get_inventory_status(self) -> List[dict]:
        """Returns inventory for dashboard rendering."""
        return [
            {
                "Part": p.name, 
                "Stock": p.stock, 
                "Threshold": p.reorder_threshold, 
                "Status": "On Order" if p.pending_order else ("Low Stock" if p.stock <= p.reorder_threshold else "In Stock")
            }
            for p in self.inventory.values()
        ]
