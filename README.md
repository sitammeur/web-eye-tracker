# 👁️ Eye Lab: Gaze Tracker API

Eye Lab is an open-source tool to create eye-tracking usability tests. It started as a final undergraduate work for the Computer Engineering student [Karine Pistili](https://www.linkedin.com/in/karine-pistili/) who made the prototype. The idea is to evolve it into a more complete and useful tool with the community's help.

The current version of the software allows users to create their usability sessions of a website, recording the webcam, screen, and mouse movements and use this information to find out where the user has been looking into the screen by using heatmaps.

## 👩‍💻 Setting up project locally

The project consists of two parts, this repository contains the backend of the application, and the frontend can be found [here](https://github.com/uramakilab/web-eye-tracker-front). Install it as well to have the full application running.

### Prerequisites

- [Python 3x](https://www.python.org/downloads/)

### 1. Create virtual environment

Before installing all dependencies and starting your Flask Server, it is better to create a Python virtual environment. You can use the [venv package](https://docs.python.org/3/library/venv.html) to create a virtual environment. To create a new virtual environment, run the following command:

```
python -m venv /path/to/new/virtual/environment
```

Then activate your environment. On Windows for example you can activate with the script:

```
name-of-event/Scripts/activate
```

### 2. Install dependencies

Install all dependencies listed on the requirements.txt with:

```
pip install -r requirements.txt
```

### 3. Run the Flask API

```
flask run
```

## Contributors ✨

The project is selected to be part of the [Google Summer of Code 2024](https://summerofcode.withgoogle.com/programs/2024/organizations/uramaki-lab) program, and [Vinícius Cavalcanti](https://github.com/hvini) is the main mentor of the project along with Marc Gonzalez Capdevila, Karine Pistili Rodrigues. The active development of the project is being done by [Sitam Meur](https://www.linkedin.com/in/sitammeur/), selected as a GSoC'24 student for the project. Here are the project details in the [GSoC'24 website](https://summerofcode.withgoogle.com/programs/2024/projects/lEPzZg7S).

To see the full list of contributions, check out the ahead commits of the "develop" branch concerning the "main" branch. Full logs of the project development can be found in the [Daily Work Progress](https://docs.google.com/document/d/1RjCnGjYYgPKvFUrN8hSjPX29aayWr6eEopeCN3QZwEQ/edit?usp=sharing) file. Hoping to see your name in the list of contributors soon! 🚀

## 🧑‍🤝‍🧑 Contributing

Anyone is free to contribute to this project. Just do a pull request with your code and if it is all good we will accept it. You can also help us look for bugs if you find anything that creates an issue.

## 📃 License

This software is under the [MIT License](https://opensource.org/licenses/MIT).

Copyright 2021 Uramaki Lab
