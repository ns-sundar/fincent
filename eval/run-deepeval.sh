 deepeval test run eval/test_e2e_intents.py 2>&1   | python -c 'import re,sys; ansi=re.compile(r"\
x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])"); [sys.stdout.write(ansi.sub("", line).replace("\r", "\n")) for line in sys.stdin]'
