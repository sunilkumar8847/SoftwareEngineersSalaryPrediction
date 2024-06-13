import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def shorten_categories(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = 'Other'
    return categorical_map

def clean_experience(x):
    if x == 'More than 50 years':
        return 50
    elif x == 'Less than 1 year':
        return 0.5
    else:
        return float(x)

def clean_education(x):
    if "Bachelor’s degree" in x:
        return "Bachelor’s degree"
    if "Master’s degree" in x:
        return "Master's degree"
    if "Professional degree" in x or "Other doctoral degree" in x:
        return "Post grad"
    return "Less than a Bachelors"

@st.cache_data
def load_data():
    df = pd.read_csv("engineer_2023.csv", encoding='ISO-8859-1')  # Adjust encoding as needed
    df = df[["Country", "EdLevel", "YearsCodePro", "Employment", "ConvertedCompYearly", "DevType"]]
    df = df.rename({"ConvertedCompYearly": "Salary"}, axis=1)
    df = df[df["Salary"].notnull()]
    df = df.dropna()
    df = df[(df["Employment"] == "Employed, full-time") | (df["Employment"] == "Student, full-time") | (df["Employment"] == "Independent contractor, freelancer, or self-employed")]
    df = df.drop("Employment", axis=1)

    country_map = shorten_categories(df.Country.value_counts(), 400)
    df["Country"] = df["Country"].map(country_map)
    df = df[df["Salary"] <= 200000]
    df = df[df["Salary"] >= 8000]
    df = df[df['Country'] != 'Other']
    df["YearsCodePro"] = df["YearsCodePro"].apply(clean_experience)
    df["EdLevel"] = df["EdLevel"].apply(clean_education)
    df = df[~df["DevType"].isin(["Other (please specify):", "Student"])]

    return df

df = load_data()

def show_explore_page():
    st.title("Explore Software Engineer Salaries")

    st.write(
        """
        ### Stack Overflow Developer Survey 2023
        """
    )

    data = df["Country"].value_counts()

    fig1, ax1 = plt.subplots(figsize=(12, 8))
    wedges, texts, autotexts = ax1.pie(data, labels=data.index, autopct="%1.1f%%", shadow=True, startangle=90,
                                       wedgeprops=dict(width=0.3), pctdistance=0.85)

    # Draw circle
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    # Equal aspect ratio ensures that pie is drawn as a circle.
    ax1.axis('equal')
    plt.tight_layout()

    st.write("""#### Number of Data from Different Countries""")
    st.pyplot(fig1)

    st.write(
    """#### Mean Salary Based On Country
    """
    )

    data = df.groupby(["Country"])["Salary"].mean().sort_values(ascending=True)
    st.bar_chart(data)

    st.write(
    """#### Mean Salary Based On Experience
    """
    )

    data = df.groupby(["YearsCodePro"])["Salary"].mean().sort_values(ascending=True)
    st.line_chart(data)

    st.write(
    """#### Mean Salary Based On Job Role
    """
    )

    data = df.groupby(["DevType"])["Salary"].mean().sort_values(ascending=True)
    st.bar_chart(data)
