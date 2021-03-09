# GitHub repository statistics scraper
# NOTE: This is very hacky and hardcoded and very likely to break in the near future
# Created by James Raphael Tiovalen (2021)

# Import libraries
import requests
from selenium import webdriver
import csv
import re

HACKERNOON_URL = "https://hackernoon.com/githubs-top-100-most-valuable-repositories-out-of-96-million-bb48caa9eb0b"
GITHUB_API_TOKEN = ""

# Confirm that page defined in the URL link still exists
resp = requests.get(HACKERNOON_URL)
if resp.status_code != 200:
    print("Failed to get top 100 most valuable GitHub repositories list.")
    exit(1)

options = webdriver.FirefoxOptions()
options.headless = True
driver = webdriver.Firefox(options=options)
driver.get(HACKERNOON_URL)

repos = (
    driver.find_element_by_class_name("ChiNr")
    .find_element_by_tag_name("ol")
    .find_elements_by_tag_name("li")
)

csvlines = []

headers = {"Authorization": f"token {GITHUB_API_TOKEN}"}


for repo in repos:
    owner_repo_pair = repo.find_element_by_tag_name("a").text[19:]

    # For status report and debugging purposes
    print(owner_repo_pair)

    base_link = f"https://api.github.com/repos/{owner_repo_pair}"
    base_resp = requests.get(base_link, headers=headers)
    if resp.status_code != 200:
        print("Failed to get repository statistics.")
        exit(1)
    resp_json = base_resp.json()

    project_name = resp_json["name"]
    project_size = resp_json["size"]
    stargazers_count = resp_json["stargazers_count"]
    watchers_count = resp_json["subscribers_count"]
    forks_count = resp_json["forks_count"]
    open_issues_count = resp_json["open_issues_count"]
    network_count = resp_json["network_count"]

    try:
        commits_count = re.search(
            "\d+$",
            requests.get(
                f"https://api.github.com/repos/{owner_repo_pair}/commits?per_page=1",
                headers=headers,
            ).links["last"]["url"],
        ).group()
    except KeyError:
        commits_count = len(
            requests.get(
                f"https://api.github.com/repos/{owner_repo_pair}/commits",
                headers=headers,
            ).json()
        )

    try:
        contributors_count = re.search(
            "\d+$",
            requests.get(
                f"https://api.github.com/repos/{owner_repo_pair}/contributors?per_page=1&anon=true",
                headers=headers,
            ).links["last"]["url"],
        ).group()
    except KeyError:
        contributors_count = len(
            requests.get(
                f"https://api.github.com/repos/{owner_repo_pair}/contributors?anon=true",
                headers=headers,
            ).json()
        )

    try:
        closed_issues_count = re.search(
            "\d+$",
            requests.get(
                f"https://api.github.com/repos/{owner_repo_pair}/issues?per_page=1&state=closed",
                headers=headers,
            ).links["last"]["url"],
        ).group()
    except KeyError:
        closed_issues_count = len(
            requests.get(
                f"https://api.github.com/repos/{owner_repo_pair}/issues?state=closed",
                headers=headers,
            ).json()
        )

    csvlines.append(
        [
            project_name,
            project_size,
            stargazers_count,
            watchers_count,
            forks_count,
            open_issues_count,
            network_count,
            commits_count,
            contributors_count,
            closed_issues_count,
        ]
    )


with open("datasets/github.csv", "w+", newline="") as csvfile:
    writer = csv.writer(
        csvfile, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL
    )
    writer.writerow(
        [
            "project_name",
            "size",
            "stars",
            "watchers",
            "forks",
            "open_issues",
            "network_count",
            "commits",
            "contributors",
            "closed_issues",
        ]
    )
    for line in csvlines:
        writer.writerow(line)