#!/usr/bin/env python3
"""
Verification script for Enterprise GenAI Platform setup
"""
import os
import sys
import json
from pathlib import Path

def check_file_exists(file_path, description):
    """Check if a file exists and report status"""
    if Path(file_path).exists():
        print(f"✅ {description}: {file_path}")
        return True
    else:
        print(f"❌ {description}: {file_path} (MISSING)")
        return False

def check_directory_structure():
    """Verify the project directory structure"""
    print("🔍 Checking project structure...")
    
    required_files = [
        # Backend files
        ("backend/app/main.py", "Backend main application"),
        ("backend/app/core/config.py", "Backend configuration"),
        ("backend/app/core/logging.py", "Backend logging setup"),
        ("backend/requirements.txt", "Backend dependencies"),
        ("backend/Dockerfile", "Backend Docker configuration"),
        ("backend/pytest.ini", "Backend test configuration"),
        
        # Frontend files
        ("frontend/package.json", "Frontend package configuration"),
        ("frontend/tsconfig.json", "TypeScript configuration"),
        ("frontend/src/App.tsx", "Frontend main component"),
        ("frontend/src/index.tsx", "Frontend entry point"),
        ("frontend/Dockerfile", "Frontend Docker configuration"),
        
        # Docker and deployment
        ("docker-compose.yml", "Docker Compose configuration"),
        ("docker-compose.dev.yml", "Development Docker Compose"),
        ("Makefile", "Build automation"),
        ("README.md", "Project documentation"),
        (".gitignore", "Git ignore rules"),
    ]
    
    all_exist = True
    for file_path, description in required_files:
        if not check_file_exists(file_path, description):
            all_exist = False
    
    return all_exist

def check_backend_config():
    """Test backend configuration loading"""
    print("\n🔍 Checking backend configuration...")
    
    try:
        # Change to backend directory to ensure .env file is found
        original_cwd = os.getcwd()
        os.chdir('backend')
        
        sys.path.insert(0, '.')
        from app.core.config import settings
        
        print(f"✅ Configuration loaded successfully")
        print(f"   - Host: {settings.HOST}")
        print(f"   - Port: {settings.PORT}")
        print(f"   - Debug: {settings.DEBUG}")
        print(f"   - Log Level: {settings.LOG_LEVEL}")
        print(f"   - LLM Model: {settings.LLM_MODEL}")
        print(f"   - LLM Base URL: {settings.LLM_BASE_URL}")
        print(f"   - Max Tokens: {settings.MAX_TOKENS}")
        
        # Check if DeepSeek API key is configured (but don't print it)
        if settings.DEEPSEEK_API_KEY and settings.DEEPSEEK_API_KEY != "your-deepseek-api-key-here":
            print(f"   - DeepSeek API Key: ✅ Configured")
        else:
            print(f"   - DeepSeek API Key: ⚠️  Not configured (using placeholder)")
        
        os.chdir(original_cwd)
        return True
    except Exception as e:
        os.chdir(original_cwd)
        print(f"❌ Configuration loading failed: {e}")
        return False

def check_frontend_config():
    """Check frontend configuration"""
    print("\n🔍 Checking frontend configuration...")
    
    try:
        with open('frontend/package.json', 'r') as f:
            package_json = json.load(f)
        
        print(f"✅ Frontend package.json loaded")
        print(f"   - Name: {package_json.get('name')}")
        print(f"   - Version: {package_json.get('version')}")
        
        # Check if required dependencies exist
        deps = package_json.get('dependencies', {})
        required_deps = ['react', 'react-dom', 'typescript', '@mui/material']
        
        missing_deps = []
        for dep in required_deps:
            if dep not in deps:
                missing_deps.append(dep)
        
        if missing_deps:
            print(f"❌ Missing dependencies: {missing_deps}")
            return False
        else:
            print(f"✅ All required dependencies present")
            return True
            
    except Exception as e:
        print(f"❌ Frontend configuration check failed: {e}")
        return False

def check_docker_config():
    """Check Docker configuration"""
    print("\n🔍 Checking Docker configuration...")
    
    try:
        # Check if docker-compose.yml is valid
        import subprocess
        result = subprocess.run(['docker-compose', 'config', '--quiet'], 
                              capture_output=True, text=True, cwd='.')
        
        if result.returncode == 0:
            print("✅ Docker Compose configuration is valid")
            return True
        else:
            print(f"❌ Docker Compose configuration error: {result.stderr}")
            return False
    except FileNotFoundError:
        print("❌ Docker Compose not found - please install Docker")
        return False
    except Exception as e:
        print(f"❌ Docker configuration check failed: {e}")
        return False

def main():
    """Main verification function"""
    print("🚀 Enterprise GenAI Platform Setup Verification")
    print("=" * 50)
    
    checks = [
        ("Project Structure", check_directory_structure),
        ("Backend Configuration", check_backend_config),
        ("Frontend Configuration", check_frontend_config),
        ("Docker Configuration", check_docker_config),
    ]
    
    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"❌ {check_name} check failed with exception: {e}")
            results.append((check_name, False))
    
    print("\n" + "=" * 50)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for check_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {check_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL CHECKS PASSED! Setup is complete.")
        print("\nNext steps:")
        print("1. Run 'make up' to start all services")
        print("2. Visit http://localhost:3000 for the frontend")
        print("3. Visit http://localhost:8000/docs for API documentation")
    else:
        print("⚠️  Some checks failed. Please review the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())