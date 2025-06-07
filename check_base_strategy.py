"""
Check the base strategy implementation to see if there's an issue
"""

# Let's look at the base strategy file content
try:
    with open('src/strategy/base_strategy.py', 'r') as f:
        content = f.read()
        print("📄 Base Strategy Content:")
        print("=" * 50)
        print(content)
except FileNotFoundError:
    print("❌ Base strategy file not found")
except Exception as e:
    print(f"❌ Error reading base strategy: {e}")