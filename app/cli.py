import argparse
import logging
import sys
from app.models.registry import ModelRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cli")

def list_models():
    registry = ModelRegistry()
    active = registry.get_all_active_models()
    print("="*60)
    print(f"{'MODEL':<20} | {'VERSION':<15} | {'TIER':<10} | {'RUNTIME'}")
    print("="*60)
    for row in active:
        tier = row.get('metadata', {}).get('tier', 'stable')
        print(f"{row['model_name']:<20} | {row['version']:<15} | {tier:<10} | {row['runtime']}")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description="InferX CLI")
    subparsers = parser.add_subparsers(dest="command")
    
    subparsers.add_parser("list", help="List active models")
    
    args = parser.parse_args()
    
    if args.command == "list":
        list_models()
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
