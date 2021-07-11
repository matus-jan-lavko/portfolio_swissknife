import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="portfolio_swissknife",
    version="0.0.1",
    author="Matus Jan Lavko",
    author_email="matusjanlavko@gmail.com",
    description="An end-to-end tool for quick portfolio analysis and sketching of investment ideas.",
    long_description_content_type="text/markdown",
    url="https://github.com/matus-jan-lavko/portfolio_swissknife",
    project_urls={
        "Bug Tracker": "https://github.com/matus-jan-lavko/portfolio_swissknife/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)