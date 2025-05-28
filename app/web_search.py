"""
Simple web search functionality - just open browser and search
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


class WebSearch:
    """Simple Chrome browser opener for Google search"""
    
    def __init__(self):
        self.driver = None
    
    def clean_message(self, message: str) -> str:
        """Remove ? characters and extra spaces"""
        if not message:
            return ""
        return ' '.join(message.replace('?', '').split()).strip()
    
    def search(self, message: str) -> bool:
        """Open Chrome, go to Google, and search the message"""
        cleaned_message = self.clean_message(message)
        
        if not cleaned_message:
            print("❌ Cannot search: message is empty")
            return False
        
        try:
            # Setup Chrome
            options = Options()
            options.add_argument('--window-size=1200,800')
            
            try:
                from webdriver_manager.chrome import ChromeDriverManager
                from selenium.webdriver.chrome.service import Service
                service = Service(ChromeDriverManager().install())
                self.driver = webdriver.Chrome(service=service, options=options)
            except ImportError:
                self.driver = webdriver.Chrome(options=options)
            
            print(f"🔍 Searching Google for: '{cleaned_message}'")

            # make uri from the cleaned message
            cleaned_message = cleaned_message.replace(' ', '+')

            url = f"https://www.duckduckgo.com/search?q={cleaned_message}"
            
            # Open Google
            self.driver.get(url)
            
            # Find and use search box
            wait = WebDriverWait(self.driver, 10000)
            
            print("✅ Search complete! Browser window is open for you to view results")
            print("🌐 Close the browser window when you're done")
            

            
        except Exception as e:
            print(f"❌ Search failed: {e}")
            return False


def search_decoded_message(message: str) -> bool:
    """
    Simple function: search a message on Google
    
    Args:
        message: The decoded message to search
        
    Returns:
        True if browser opened successfully
    """
    print(f"\n📝 Message: '{message}'")
    
    if not message or message.replace('?', '').strip() == '':
        print("⚠️  Nothing to search")
        return False
    
    searcher = WebSearch()
    return searcher.search(message)


# Example usage
if __name__ == "__main__":
    # Test search
    test_message = "HELLO WORLD"
    search_decoded_message(test_message)
    # keep application running
    input("Press Enter to exit...")
    if search_decoded_message(test_message):
        print("Search completed successfully.")
    else:
        print("Search failed.")