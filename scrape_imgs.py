import base64
import os
import time

import requests
from selenium import webdriver
from selenium.webdriver import FirefoxOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


def download_images(query, num_images, output_folder):
    # Creating a webdriver instance
    opts = FirefoxOptions()
    opts.add_argument("--headless")
    driver = webdriver.Firefox(options=opts)

    # Maximize the screen
    driver.maximize_window()

    # Open Google Images in the browser
    driver.get("https://images.google.com/")

    # Refuse all cookies (assumes French Google Homepage)
    WebDriverWait(driver, 9).until(
        EC.element_to_be_clickable((By.XPATH, "//div[text()='Tout refuser']"))
    ).click()

    # Submit query
    search_box = driver.find_element(By.TAG_NAME, "textarea")
    search_box.send_keys(query)
    search_box.submit()
    time.sleep(3)

    # Get image URLs
    img_elements = driver.find_elements(By.CSS_SELECTOR, ".rg_i.Q4LuWd")
    img_urls = [e.get_attribute("src") for e in img_elements if e.get_attribute("src")]

    # Download images
    for i, img_url in enumerate(img_urls[:num_images]):
        img_name = f"{query}_image_{i+1}.jpg"
        img_path = os.path.join(output_folder, img_name)

        # Check if the input is a Base64-encoded string
        if img_url.startswith("data:image"):
            # Extract the Base64 data from the input string
            base64_data = img_url.split(",")[1]

            # Decode the Base64 string
            image_data = base64.b64decode(base64_data)

            with open(img_path, "wb") as img_file:
                img_file.write(image_data)

        # Check if the input is a URL
        elif img_url.startswith(("http://", "https://")):
            # Download the image using requests
            with open(img_path, "wb") as img_file:
                img_file.write(requests.get(img_url).content)

        print(f"Downloaded {img_name}")

    # ### Fast DEBUGGING
    # driver.save_screenshot("screenshot.png")

    driver.close()


if __name__ == "__main__":
    # Params
    list_queries = ["pizza picnic", "pizza restaurant", "pizza"]
    num_images_to_download = 10
    output_folder_path = "./imgs"

    # Ensure the output folder exists
    os.makedirs(output_folder_path, exist_ok=True)

    for search_query in list_queries:
        download_images(search_query, num_images_to_download, output_folder_path)
