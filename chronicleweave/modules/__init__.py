# chronicleweave/modules/__init__.py
#
# Intentionally empty: each module is imported on demand by its consumer (pipeline.py
# and the tests). Eagerly importing every module here forced numpy/pydub/google-genai
# to load even for callers that only need a single helper.
