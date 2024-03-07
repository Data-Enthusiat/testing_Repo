from Preprocessing_Parsing import ResumeProcessor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from JD import Job_Description
class Matching:
    def cal_cosine_similarity(self, resume_skills, job_description_skills, corpus, threshold=15):
        # If job description skills not present in resume
        jd = Job_Description()
        skills_not_in_resume = jd.find_not_in_resume(resume_skills, job_description_skills)
        missing_skills = ", ".join(skills_not_in_resume)
        # Creating a vectorizer
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(corpus)
        # Calculating cosine similarity
        resume_text, job_description_text = " ".join(resume_skills), " ".join(job_description_skills)
        similarity = cosine_similarity(X[0], X[1])
        score = round(similarity[0][0] * 100, 2)
        missing_skill = dict(enumerate(skills_not_in_resume))
        return score, missing_skill

