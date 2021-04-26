# Created by James Raphael Tiovalen (2021)

#%%
import numpy as np
from scipy.special import polygamma
from scipy.stats import gamma
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup, SoupStrainer
import pandas as pd
from sklearn import linear_model

#%%
### Question 1b
k = np.arange(1, 21, 1)
n = np.array([3, 1, 14, 15, 10, 11, 19, 21, 22, 18, 9, 12, 8, 6, 3, 2, 3, 1, 1, 1])

l = np.log(n.dot(k) / n.sum())
s = n.dot(np.log(k)) / n.sum()

print("Total number of samples: " + str(n.sum()))
print("Log of average of all samples: " + str(l))
print("Average of all log of samples: " + str(s))

alpha_covid = 1764 / 95

THRESHOLD = 1e-1000
prev_alpha = n.dot(k) / n.sum()
current_alpha = prev_alpha
running = True
number_of_iterations = 0
while running:
    current_alpha -= ((polygamma(0, prev_alpha) - np.log(prev_alpha) + l - s) / (polygamma(1, prev_alpha)))
    number_of_iterations += 1
    if (np.abs(current_alpha - prev_alpha) <= THRESHOLD):
        running = False
    prev_alpha = current_alpha
print("MLE of SARS alpha: " + str(current_alpha))
print("Number of iterations: " + str(number_of_iterations))

beta = current_alpha / (n.dot(k) / n.sum())
print("MLE of SARS beta: " + str(beta))

#%%
# Question 1d
x = np.linspace(0, 10, 10000)
theta_values = np.arange(1, 5, 1)
y1 = gamma.pdf(x, a=2, loc=0, scale=1/theta_values[0])
y2 = gamma.pdf(x, a=2, loc=0, scale=1/theta_values[1])
y3 = gamma.pdf(x, a=2, loc=0, scale=1/theta_values[2])
y4 = gamma.pdf(x, a=2, loc=0, scale=1/theta_values[3])

plt.plot(x, y1, "r", label=r"""$\alpha=2, \beta=1$""")
plt.plot(x, y2, "g", label=r"""$\alpha=2, \beta=2$""")
plt.plot(x, y3, "b", label=r"""$\alpha=2, \beta=3$""")
plt.plot(x, y4, "purple", label=r"""$\alpha=2, \beta=4$""")

plt.xlim([0,10])
plt.legend()
plt.show()

#%%
# Question 2a
url = "https://www.moh.gov.sg/covid-19/past-updates"
page_keyword = "New Cases of COVID-19 Infection Confirmed"
pdf_keyword = "annex-c"
relevant_links = []

print("Getting the URL links of all the relevant PDF documents by scraping MOH's website...")

response = requests.get(url).content

for toplink in BeautifulSoup(response, features="lxml", parse_only=SoupStrainer("a")):
    try:
        if page_keyword in toplink.string and hasattr(toplink, "href"):
            relevant_links.append(toplink["href"])
    except TypeError as e:
        pass

relevant_links = relevant_links[:34]
pdf_links = []

for link in relevant_links:
    content = requests.get(link).content
    for sublink in BeautifulSoup(content, features="lxml", parse_only=SoupStrainer("a")):
        if hasattr(sublink, "href"):
            try:
                if pdf_keyword in sublink["href"]:
                    pdf_links.append(sublink["href"])
            except KeyError as e:
                pass

assert len(pdf_links) == len(relevant_links)

print("Starting to download all the relevant PDF documents...")

for i in range(len(pdf_links)):
    pdf = requests.get(pdf_links[i])
    with open(f"{i}.pdf", "wb") as f:
        f.write(pdf.content)

#%%
# Question 2c
observed_g_values = [1, 2, 2, 7, 22, 4, 1, 1, 3, 1, 3, 4, 1, 1, 2, 2, 2, 2, 1, 3, 13, 16, 2, 4, 3, 1, 1]

old_alpha = 2
old_beta = 19 / 42

new_alpha = old_alpha + (len(observed_g_values) * (1764 / 95))
new_beta = old_beta + np.sum(observed_g_values)

beta_covid = gamma.mean(a=new_alpha, loc=0, scale=1/new_beta)

# %%
# Question 2e
x = np.linspace(0, 10, 10000)
for i in np.arange(0.01, 4.01, 0.01):
    alpha = 2 + (len(observed_g_values) * (1764 / 95))
    beta = i + np.sum(observed_g_values)
    y = gamma.pdf(x, a=alpha, loc=0, scale=1/beta)
    plt.plot(x, y, label=r"""$\alpha={a}, \beta={b}$""".format(a=alpha, b=beta))

plt.show()

# %%
# Question 3b

pd.set_option('display.max_rows', None)
df = pd.read_csv("data.csv", parse_dates=["dateRep"], dayfirst=True)
usa = df.countryterritoryCode.str.fullmatch("USA", na=False)
march = df.month.astype(str).str.fullmatch("3", na=False)
target = df[usa].sort_values("dateRep")
target["cases_cum_sum"] = target["cases"].cumsum()
z = target[61:92]["cases_cum_sum"].to_numpy()
y = np.log(z)
x = np.arange(1, 32, 1)

x_mean = np.mean(x)
y_mean = np.mean(y)

m = np.dot(y_mean - y, x_mean - x) / np.sum(np.power(x_mean - x, 2))
c = y_mean - m * x_mean
print(f"r = {m}")
print(f"n0 = {np.exp(c)}")

model = linear_model.LinearRegression()
model.fit(x.reshape(-1, 1), y)
print(f"r = {model.coef_[0]}")
print(f"n0 = {np.exp(model.intercept_)}")

r = m

# %%
# Question 4a

R_0 = (1 + (r / beta_covid)) ** alpha_covid
print("Basic reproduction number of COVID-19 is approximately: " + str(R_0))

# %%
