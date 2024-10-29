#!/usr/bin/env python3
import os
import getpass
import subprocess
from pathlib import Path
import re

def validate_api_key(key: str) -> bool:
    """
    Validate that the API key matches common formats.
    Basic validation to prevent obviously invalid keys.
    """
    # Remove whitespace
    key = key.strip()
    
    # Check if key is empty
    if not key:
        return False
    
    # Check for common API key patterns
    # Most API keys are alphanumeric and may include some special characters
    valid_pattern = re.compile(r'^[A-Za-z0-9\-_\.]+$')
    return bool(valid_pattern.match(key))

def update_shell_rc(env_vars: dict) -> str:
    """
    Update shell RC file (.bashrc or .zshrc) with environment variables
    Returns the path to the RC file that was updated
    """
    # Determine which shell RC file to use
    shell = os.environ.get('SHELL', '/bin/bash')
    home = str(Path.home())
    
    if 'zsh' in shell:
        rc_file = os.path.join(home, '.zshrc')
    else:
        rc_file = os.path.join(home, '.bashrc')
    
    # Create backup of RC file
    backup_file = f"{rc_file}.backup"
    try:
        subprocess.run(['cp', rc_file, backup_file], check=True)
        print(f"Created backup of {rc_file} at {backup_file}")
    except subprocess.CalledProcessError:
        print(f"Warning: Could not create backup of {rc_file}")
    
    # Add or update environment variables
    with open(rc_file, 'a') as f:
        f.write('\n# API Keys added by setup script\n')
        for key, value in env_vars.items():
            f.write(f'export {key}="{value}"\n')
    
    return rc_file

def source_rc_file(rc_file: str, env_vars: dict) -> None:
    """
    Source the RC file to apply changes to the current environment
    """
    verify_script_path = '/tmp/verify_api_keys.sh'
    
    try:
        # Export variables to current environment
        for key, value in env_vars.items():
            os.environ[key] = value
        
        # Create verification script content with correct variable expansion
        verify_script = '''#!/bin/bash
source "{rc_file}"
'''.format(rc_file=rc_file)

        # Add verification for each key
        for key in env_vars.keys():
            verify_script += f'''
if [ -n "${key}" ]; then
    echo "{key} is set to: ${{{key}}}"
else
    echo "{key} is not set"
    exit 1
fi
'''
        
        # Write verify script to temporary file
        with open(verify_script_path, 'w') as f:
            f.write(verify_script)
        os.chmod(verify_script_path, 0o755)
        
        # Run verification
        result = subprocess.run(['/bin/bash', verify_script_path], 
                              capture_output=True, 
                              text=True)
        
        if result.returncode == 0:
            print("\nAPI keys have been successfully set in the current environment.")
            print(f"\nPlease now run \"source {rc_file}\" to complete configuration.")
        else:
            print("Verification failed. Script output:")
            print(result.stdout)
            print(result.stderr)
            raise subprocess.CalledProcessError(result.returncode, verify_script_path)
            
    except Exception as e:
        print(f"\nWarning: Could not automatically apply changes to current session: {str(e)}")
        print(f"Please run 'source {rc_file}' manually to apply changes.")
    finally:
        # Clean up
        if os.path.exists(verify_script_path):
            os.remove(verify_script_path)

def main():
    print("API Key Setup Script")
    print("===================")
    print("\nThis script will help you set up your API keys as environment variables.")
    
    api_keys = {
        "TOGETHER_API_KEY": "Together API Key",
        "ELSEVIER_API_KEY": "Elsevier API Key"
    }
    
    collected_keys = {}
    
    # Collect API keys
    for env_var, description in api_keys.items():
        while True:
            key = getpass.getpass(f"\nEnter your {description}: ")
            if validate_api_key(key):
                collected_keys[env_var] = key
                break
            else:
                print("Invalid API key format. Please try again.")
    
    # Update shell RC file and source changes
    try:
        rc_file = update_shell_rc(collected_keys)
        source_rc_file(rc_file, collected_keys)
    except Exception as e:
        print(f"\nError updating shell configuration: {str(e)}")
        print("Please add the following lines to your shell configuration file manually:")
        for key, value in collected_keys.items():
            print(f'export {key}="{value}"')

if __name__ == "__main__":
    main()