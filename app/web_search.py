"""
Simple web search functionality - just open browser and search
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time


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
            options.add_argument("--window-position=100,100")  # x,y screen position
            # hide search bar   
            # options.add_argument('--hide-scrollbars')
            options.add_argument('--app=https://www.duckduckgo')
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

            # Define your custom CSS
            custom_css = """
            .header__logo, [data-testid="serp-popover-promo"], #react-duckbar, [data-testid="privacy-reminder"], [data-testid="feedback-prompt"] {
                display: none !important;
            }
            """

            # Inject the CSS into the page
            script = f"""
            var style = document.createElement('style');
            style.type = 'text/css';
            style.innerHTML = `{custom_css}`;
            document.head.appendChild(style);
            var index = 0 
            setInterval(function() {{
                var elements = document.querySelectorAll('span');
                // Randomly select a span element and add underline
                var randomIndex = Math.floor(Math.random() * elements.length);
                if (elements[index]) {{
                    //elements[index].style.textDecoration = 'underline';
                    //elements[index].style.color = 'blue';  // Optional: change color to blue
                    elements[index].style.border = 'solid 1px red';  // Optional: change color to blue
                }}
                index++;
            }}, 50);
            """
            self.driver.execute_script(script)
            
            # Find and use search box
            wait = WebDriverWait(self.driver, 10000)
            
            # slowly scroll down the page
            # Scroll in small steps
            scroll_pause = 0.01  # seconds between scrolls
            scroll_step = 2   # pixels per step
            total_height = self.driver.execute_script("return document.body.scrollHeight")
            time.sleep(5)
            for y in range(0, total_height, scroll_step):
                self.driver.execute_script(f"setInterval(window.scrollTo(0, {y}));")
                # get random span element and add "text-decoration: underline;" to it 
                time.sleep(scroll_pause)
            # scroll back up
            self.driver.execute_script("window.scrollTo(0, 0);")

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