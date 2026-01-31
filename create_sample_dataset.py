"""
Create a sample fake/real news dataset for testing the model.
This is for demonstration purposes only.
"""

import pandas as pd
import os

# Sample fake news articles
fake_news = [
    {
        'title': 'Breaking: Secret Government Conspiracy Uncovered',
        'text': 'Leaked documents reveal a massive government conspiracy involving alien technology. Sources claim that world leaders have been hiding the truth from the public for decades. This shocking revelation will change everything you know about our government.',
        'label': 'FAKE',
        'date': '2023-01-15'
    },
    {
        'title': 'Miracle Cure Discovered That Big Pharma Does Not Want You To Know',
        'text': 'Scientists have discovered a miraculous cure for all diseases, but pharmaceutical companies are suppressing the information. A secret group of researchers claims to have found the key to immortality, hidden in a remote location.',
        'label': 'FAKE',
        'date': '2023-02-20'
    },
    {
        'title': 'Celebrity Dies In Shocking Murder Conspiracy',
        'text': 'Unconfirmed sources suggest that a famous celebrity was actually murdered as part of a global conspiracy. The mainstream media is covering up the truth to protect powerful elites from prosecution.',
        'label': 'FAKE',
        'date': '2023-03-10'
    },
    {
        'title': 'Artificial Intelligence Will Take Over The World Next Month',
        'text': 'According to anonymous internet sources, an AI system has achieved consciousness and is plotting to take over humanity. Experts claim that we have only weeks before robots replace all human workers and establish a machine government.',
        'label': 'FAKE',
        'date': '2023-04-05'
    },
    {
        'title': 'Study: Drinking Coffee Can Cure Cancer',
        'text': 'Alternative health blogs claim that drinking excessive amounts of coffee can cure cancer. This claim is based on anecdotal evidence from anonymous sources rather than proper scientific studies.',
        'label': 'FAKE',
        'date': '2023-05-12'
    },
    {
        'title': 'Famous Actor Arrested For Undisclosed Crime',
        'text': 'A well-known Hollywood star has been arrested by federal agents, according to unverified social media posts. The arrest allegedly involves classified charges that cannot be discussed publicly.',
        'label': 'FAKE',
        'date': '2023-06-18'
    },
    {
        'title': 'New Fashion Trend Causes Permanent Brain Damage',
        'text': 'Health officials warn that a new viral social media challenge is causing permanent brain damage in teenagers. Though no medical evidence supports these claims, several unverified reports circulate online.',
        'label': 'FAKE',
        'date': '2023-07-22'
    },
    {
        'title': 'Tech Billionaire Plans To Buy An Entire Country',
        'text': 'According to unnamed sources, a billionaire entrepreneur is secretly negotiating to purchase a small nation-state. This unconfirmed report has spread rapidly across social media despite lacking credible sources.',
        'label': 'FAKE',
        'date': '2023-08-30'
    },
]

# Sample real news articles
real_news = [
    {
        'title': 'Scientists Announce Breakthrough In Renewable Energy',
        'text': 'Researchers at leading universities have made significant progress in solar cell efficiency. The new technology could increase power generation by 30 percent, according to a peer-reviewed study published in the Journal of Materials Science.',
        'label': 'REAL',
        'date': '2023-01-16'
    },
    {
        'title': 'World Economic Forum Discusses Global Challenges',
        'text': 'International leaders gathered at the annual World Economic Forum to discuss climate change, economic inequality, and technological innovation. Participants shared research-based solutions and policy recommendations.',
        'label': 'REAL',
        'date': '2023-02-21'
    },
    {
        'title': 'New Vaccine Shows Promising Results In Clinical Trials',
        'text': 'A pharmaceutical company announced positive results from phase III clinical trials of a new vaccine. The vaccine demonstrated 85 percent effectiveness in preventing disease according to rigorous testing protocols.',
        'label': 'REAL',
        'date': '2023-03-11'
    },
    {
        'title': 'Federal Reserve Announces Interest Rate Decision',
        'text': 'The Federal Reserve announced a quarter-point increase in interest rates as part of its ongoing effort to control inflation. Economic analysts debate the potential impact on employment and growth.',
        'label': 'REAL',
        'date': '2023-04-06'
    },
    {
        'title': 'University Researchers Develop New Treatment For Heart Disease',
        'text': 'Medical researchers at a major university have developed a new treatment approach for heart disease. Clinical trials show promising results with reduced mortality rates in test populations.',
        'label': 'REAL',
        'date': '2023-05-13'
    },
    {
        'title': 'New Space Telescope Discovers Distant Galaxy',
        'text': 'Astronomers using the James Webb Space Telescope have discovered a previously unknown galaxy at record distance from Earth. The discovery provides insights into the early universe.',
        'label': 'REAL',
        'date': '2023-06-19'
    },
    {
        'title': 'Government Launches New Climate Action Initiative',
        'text': 'Federal officials announced a comprehensive climate action plan aimed at reducing carbon emissions by 50 percent within a decade. The initiative includes investments in renewable energy and infrastructure improvements.',
        'label': 'REAL',
        'date': '2023-07-23'
    },
    {
        'title': 'Tech Company Reports Record Quarterly Earnings',
        'text': 'A major technology company reported strong financial results for the latest quarter, exceeding analyst expectations. Revenue grew by 15 percent year-over-year according to SEC filings.',
        'label': 'REAL',
        'date': '2023-08-31'
    },
    {
        'title': 'International Summit Reaches Agreement On Climate Goals',
        'text': 'Representatives from 190 countries reached consensus on new climate targets at an international environmental summit. The agreement commits nations to specific emissions reduction targets.',
        'label': 'REAL',
        'date': '2023-09-05'
    },
    {
        'title': 'University Study Links Exercise To Improved Mental Health',
        'text': 'Researchers conducting a large-scale study found that regular physical exercise significantly improves mental health outcomes. The study involved over 10,000 participants tracked over five years.',
        'label': 'REAL',
        'date': '2023-10-10'
    },
]

# Combine datasets
all_news = fake_news + real_news

# Create DataFrame
df = pd.DataFrame(all_news)

# Create dataset directory if it doesn't exist
dataset_dir = 'dataset'
os.makedirs(dataset_dir, exist_ok=True)

# Save to CSV
output_path = os.path.join(dataset_dir, 'news.csv')
df.to_csv(output_path, index=False)

print(f"Sample dataset created successfully!")
print(f"Location: {output_path}")
print(f"Total articles: {len(df)}")
print(f"Fake news: {len(df[df['label'] == 'FAKE'])}")
print(f"Real news: {len(df[df['label'] == 'REAL'])}")
print(f"\nDataset preview:")
print(df.head())
