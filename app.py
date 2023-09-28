import streamlit as st
from PIL import Image

from model import model



def main():
	with st.sidebar:
		choice = st.radio("Choose:",
					(" :blue[_Sentiment Analysis for Movie Review_]", "About me"))

	if choice == " :blue[_Sentiment Analysis for Movie Review_]":
		model()

	else:
		st.write(" ## :star2: Aniket Narpatraj Bafna :star2:")
		st.write(" #### :male-technologist: Data Scientist :male-technologist:")
		img = Image.open("aniket.jpeg")
		st.image(img, caption="Aniket Bafna", width=300)

		st.text("""
		Highly motivated and results-driven Self Taught Data Scientist with over 2 year of hands-on experience in the field of Machine Learning, 
		Natural Language Processing (NLP), Data Visualization, Data Processing, Tableau, Power BI, Statistics, and more.

		I am passionately dedicated to harnessing the power of data to derive actionable insights and have an unwavering commitment to professional 
		growth, adaptability, and collaborative teamwork. I bring a strong analytical mindset and a passion for continuous learning to every project.

		As a highly adaptable and growth-oriented individual, I am keen to seize new opportunities that allow me to broaden my skill set, learn 
		from diverse experiences, and contribute to impactful data-driven initiatives. My collaborative spirit and insatiable curiosity make me 
		a valuable asset in the fast-evolving world of data science, where innovation and problem-solving are paramount. I am excited to embark on 
		new ventures that push the boundaries of what is possible through data-driven insights and solutions.

		I love meeting people working on exciting things. If there is any suitable role for me, don't hesitate. 
		I am open to communication on all channels. 
		Let's discuss.
""")
		socials = ["LinkedIn", "Github", "Gmail"]
		linkedin = "http://www.linkedin.com/in/aniket-bafna"
		github = "https://github.com/AniketBafna"
		mail = "aniketbafna0@gmail.com"
		Tableau = ""
		with st.expander("Links to all my Socials"):
			a = st.selectbox("Socials", socials)
			if a =="LinkedIn":
				st.write(linkedin)
			elif a =="Github":
				st.write(github)
			elif a=="Gmail":
				st.write(mail)

main()
