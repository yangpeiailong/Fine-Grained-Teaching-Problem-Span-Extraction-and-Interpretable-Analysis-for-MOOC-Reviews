from selenium import webdriver
from bs4 import BeautifulSoup
from openpyxl import Workbook
from selenium.webdriver.common.by import By
import time
import os

# Crawl MOOC course reviews and export to Excel
# Function: Extract user reviews, ratings, like counts, etc. from icourse163.org
if __name__ == '__main__':
    # Course name for output file naming
    course_name = "人工智能原理"

    # Initialize Excel workbook and worksheet
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = 'course_reviews'

    # Write Excel header
    headers = ['user_nickname', 'review_content', 'review_time', 'like_count', 'course_term', 'rating']
    worksheet.append(headers)

    # Initialize Edge driver
    driver = webdriver.Edge()
    driver.implicitly_wait(10)

    # Target course URL
    course_url = 'https://www.icourse163.org/course/PKU-1002188003?from=searchPage&outVendor=zw_mooc_pcssjg_'
    driver.get(course_url)

    # Click review tab
    review_button = driver.find_element(By.ID, "review-tag-button")
    review_button.click()
    time.sleep(2)

    # Pagination setup
    page_num = 1
    max_pages = 28

    # Start crawling
    while page_num <= max_pages:
        print(f"Crawling page {page_num}...")
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')

        # Locate all review items
        review_items = soup.find_all('div', {
            'class': 'ux-mooc-comment-course-comment_comment-list_item_body'
        })

        for item in review_items:
            # Extract user nickname
            user_name_elem = item.find("a", {
                "class": "primary-link ux-mooc-comment-course-comment_comment-list_item_body_user-info_name"
            })
            user_nickname = user_name_elem.text.strip() if user_name_elem else "Unknown User"

            # Extract review time (remove prefix text)
            publish_time_elem = item.find('div', {
                'class': 'ux-mooc-comment-course-comment_comment-list_item_body_comment-info_time'
            })
            review_time = publish_time_elem.text[4:].strip() if publish_time_elem else "Unknown Time"

            # Extract course term
            course_term_elem = item.find('div', {
                'class': 'ux-mooc-comment-course-comment_comment-list_item_body_comment-info_term-sign'
            })
            course_term = course_term_elem.text.replace(" ", "").replace("\n", "").strip() if course_term_elem else "Unknown Term"

            # Extract like count
            like_elem = item.find('span', {
                'class': 'ux-mooc-comment-course-comment_comment-list_item_body_comment-info_actions_vote'
            })
            like_count = like_elem.text.strip() if like_elem else '0'

            # Extract review content
            content_elem = item.find('div', {'class': 'ux-mooc-comment-course-course-comment_comment-list_item_body_content'})
            review_content = content_elem.text.strip() if content_elem else ''

            # Extract rating (number of stars)
            rating_stars = item.find_all('i', {'class': 'star icon-rating-favorite'})
            rating = len(rating_stars)

            # Write one row of data
            row_data = [user_nickname, review_content, review_time, like_count, course_term, rating]
            worksheet.append(row_data)

        # Go to next page
        if page_num < max_pages:
            try:
                next_page_btn = driver.find_element(By.CLASS_NAME, "ux-pager_btn__next")
                next_page_btn.click()
                time.sleep(2)
            except:
                print("No more pages. Crawling stopped.")
                break
        page_num += 1

    # ===================== Relative Path Configuration =====================
    # Current script directory: ./codes
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Output directory: ../raw_datas
    output_dir = os.path.join(base_dir, "..", "raw_datas")

    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)

    # Save Excel file
    excel_filename = f"{course_name}_reviews.xlsx"
    save_path = os.path.join(output_dir, excel_filename)
    workbook.save(save_path)
    print(f"Excel saved to: {save_path}")

    # Save URL to TXT file
    txt_filename = f"{course_name}.txt"
    txt_save_path = os.path.join(output_dir, txt_filename)
    with open(txt_save_path, 'w', encoding='utf-8') as f:
        f.write(course_url)
    print(f"TXT saved to: {txt_save_path}")

    # Close browser
    driver.quit()