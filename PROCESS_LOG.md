# Process Log

## AI tools used

I used Claude, specifically the Claude Opus 5.5 version, to help delegate what subject or topic to use for this project and I landed on a system that is about anatomy and physiology open source education book for premed students.

## Development narrative

I wanted to make this related to the medical field since I am a project manager at Kaiser Permanente. Besides a few minor hiccups within the testing and production scope of the project, there was a notable blocker I ran into where I had performed an overnight test of 30 questions through the entire system twice to with one having research deployed with Nemotron and the other on Gemini with scored Ragas as it was part of the requirements. Another blocker I reached was there being a Gemini rate limit and how the slow plan led to the hedge router.

## Verification narrative

There was an implementation of 70 tests to run, the RAGAS run, measuring the cold start website metrics, and as well as manual checks on the website.I also tested to see if I could ask questions out of scope of anatomy and physiology and out of scope of the textbook.

## What you learned

I learned that depending on the model and agent assistant you use, no matter how complex, sophisticated, and astonishing the results are, there is always room for error. Such as interpreting beyond the code and seeing if what is inputted and output makes sense not only in the scope of the code, but also results, for example as an API key was exposed during code work and had to be deleted and regenerated as it was attached to the MCP URL
