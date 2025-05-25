import os

# Create documentation directories
dirs = [
    'docs/getting_started',
    'docs/user_guide',
    'docs/api',
    'docs/development',
    'docs/_static'
]

for dir_path in dirs:
    os.makedirs(dir_path, exist_ok=True)
    print(f"Created directory: {dir_path}")

# Create .nojekyll file to ensure GitHub Pages builds properly
with open('docs/.nojekyll', 'w') as f:
    f.write('')

print("\nDocumentation structure created successfully!")
