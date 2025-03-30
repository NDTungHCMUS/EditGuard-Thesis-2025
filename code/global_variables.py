# ----- VN START -----
TEST_CONFIG_FILE = "options/test_editguard.yml"

import yaml
def load_config(path=TEST_CONFIG_FILE):
    with open(path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

# Global configuration variable
TEST_CONFIG = load_config()
# ----- VN END ----