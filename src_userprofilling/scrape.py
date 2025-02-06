from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.common.wait import WebDriverWait # type: ignore
from selenium.webdriver.support import expected_conditions as EC
import time

# Path to your WebDriver (Change this to your local path)
CHROME_DRIVER_PATH = r"C:\Users\ABISHEKRS\Downloads\chromedriver-win64\chromedriver-win64\chromedriver.exe"
def scrape_instagram(user):
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Run in headless mode (no GUI)
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("window-size=1920x1080")
    options.add_argument("--disable-notifications")

    service = Service(CHROME_DRIVER_PATH)
    driver = webdriver.Chrome(service=service, options=options)

    try:
        url = f"https://www.instagram.com/{user}/"
        driver.get(url)

        # Wait until the profile page is fully loaded
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )

        time.sleep(3)  # Allow some extra time for loading

        # Extract profile picture URL
        profile_pic_element = driver.find_element(By.XPATH, "//img[@alt='Instagram profile picture']"
)
        profile_pic_url = profile_pic_element.get_attribute("src")

        # Extract bio section
        try:
            bio_element = driver.find_element(By.XPATH, "//div[contains(@class, 'x1qjc9v5')]/span")
            bio_text = bio_element.text
        except:
            bio_text = "Bio not found"

        # Extract followers, following, and post count
        stats = driver.find_elements(By.XPATH, "//ul/li//span[@class='x193iq5w x1qjc9v5 x1q0g3np']")
        followers = stats[0].text if len(stats) > 0 else "N/A"
        following = stats[1].text if len(stats) > 1 else "N/A"
        posts = stats[2].text if len(stats) > 2 else "N/A"

        data = {
            "username": user,
            "followers": followers,
            "following": following,
            "posts": posts,
            "bio": bio_text,
            "profile_picture": profile_pic_url,
        }

        return data

    except Exception as e:
        return {"error": str(e)}

    finally:
        driver.quit()

if __name__ == "__main__":
    username = input("Enter Instagram username: ")
    user_data = scrape_instagram(username)
    print(user_data)
