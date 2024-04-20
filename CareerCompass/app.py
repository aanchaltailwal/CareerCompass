import streamlit as st
import pickle
import re
import nltk
import requests

nltk.download('punkt')
nltk.download('stopwords')

#loading models
clf = pickle.load(open('clf.pkl','rb'))
tfidfd = pickle.load(open('tfidf.pkl','rb'))

def clean_resume(resume_text):
    clean_text = re.sub('http\\S+\\s*', ' ', resume_text)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('#\\S+', '', clean_text)
    clean_text = re.sub('@\\S+', '  ', clean_text)
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', r' ', clean_text)
    clean_text = re.sub('\\s+', ' ', clean_text)
    return clean_text

# Function to fetch LinkedIn jobs
def fetch_linkedin_jobs(api_key, field, geoid, page):
    url = "https://api.scrapingdog.com/linkedinjobs/"
    params = {
        "api_key": api_key,
        "field": field,
        "geoid": geoid,
        "page": page
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if isinstance(data, list):
            return data
        else:
            return data.get("jobs", [])
    else:
        return []

# web app
def main():

  app_container = st.container()

  # Add background image style to the container
  with app_container:
    st.write("""
      <style>
        .stApp {
          background-image: url("https://png.pngtree.com/thumb_back/fh260/background/20210918/pngtree-irregular-triangular-low-poly-style-cyan-background-image_902896.png");
          background-size: cover;
        }
      </style>
      """, unsafe_allow_html=True)

    st.title("Job Recommendation System")
    uploaded_file = st.file_uploader('Upload Resume', type=['txt','pdf'])

    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            # If UTF-8 decoding fails, try decoding with 'latin-1'
            resume_text = resume_bytes.decode('latin-1')

        cleaned_resume = clean_resume(resume_text)
        input_features = tfidfd.transform([cleaned_resume])
        prediction_id = clf.predict(input_features)[0]
        

        # Map category ID to category name
        category_mapping = {
            15: "Java Developer",
            23: "Testing",
            8: "DevOps Engineer",
            20: "Python Developer",
            24: "Web Designing",
            12: "HR",
            13: "Hadoop",
            3: "Blockchain",
            10: "ETL Developer",
            18: "Operations Manager",
            6: "Data Science",
            22: "Sales",
            16: "Mechanical Engineer",
            1: "Arts",
            7: "Database",
            11: "Electrical Engineering",
            14: "Health and fitness",
            19: "PMO",
            4: "Business Analyst",
            9: "DotNet Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",
            21: "SAP Developer",
            5: "Civil Engineer",
            0: "Advocate",
        }

        category_name = category_mapping.get(prediction_id, "Unknown")

        st.write("Predicted Category:", category_name)

        # Fetch LinkedIn jobs based on recommended job role
        st.header("Recommended LinkedIn Jobs")
        jobs = fetch_linkedin_jobs("662132f5ce0c211738e0d20f", category_name, "102713980", "1")
        if jobs:
            for job in jobs:
                st.write("Job Position:", job["job_position"])
                st.write("Company Name:", job["company_name"])
                st.write("Location:", job["job_location"])
                st.write("Posting Date:", job["job_posting_date"])
                st.write("Job Link:", job["job_link"])
                st.write("---")
        else:
            st.warning("No jobs found. Please try again.")

# python main
if __name__ == "__main__":
    main()
