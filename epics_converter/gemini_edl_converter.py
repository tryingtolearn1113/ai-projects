import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load your Gemini API key
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

def read_file(filepath):
    """Reads a file and returns its text."""
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return ""

def main():
    print("=== EPICS EDL to Python AI Converter ===\n")

    # 1. Load your perfect examples (The "Training Data")
    print("Reading your examples...")
    example_edl_left = read_file("examples/SurvLeft.edl")
    example_edl_right = read_file("examples/SurvRight.edl")
    example_py = read_file("examples/pls_monitor_end2.py")

    # 2. Setup the AI Expert
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0.1 # Low temperature so it doesn't get overly creative
    )

    # 3. Create the Instruction Template
    prompt = ChatPromptTemplate.from_template("""
You are an expert EPICS control system software engineer. 
Your job is to convert old .edl / .medm files into modern Python Tkinter/PyDM scripts.

Here is an example of what an old EDL looks like (Left Screen):
{left_edl}

Here is another example of an old EDL (Right Screen):
{right_edl}

Here is the PERFECT modern Python script the user wrote to replace them. 
Notice how they combined the screens, used classes, and structured the code:
{python_example}

Now, please convert the following NEW .edl file into a Python script using the EXACT SAME STYLE as the perfect Python script above.
Output ONLY the Python code. Do not include markdown formatting or explanations.

NEW EDL TO CONVERT:
{new_edl}
""")

    # 4. Build the conversion pipeline
    chain = prompt | llm | StrOutputParser()

    # 5. Ask the user for a file to convert
    target_file = input("\nEnter the path of the .edl file you want to convert: ")
    
    if not os.path.exists(target_file):
        print("File not found! Please check the path.")
        return

    print("\nReading target file...")
    new_edl_content = read_file(target_file)

    print("\nAI is thinking... (This might take a minute, it is reading a lot of code!)")
    
    # 6. Run the AI
    result = chain.invoke({
        "left_edl": example_edl_left,
        "right_edl": example_edl_right,
        "python_example": example_py,
        "new_edl": new_edl_content
    })

    # 7. Clean up the output and save it
    result = result.replace("```python", "").replace("```", "").strip()
    
    output_filename = target_file.replace(".edl", "_converted.py")
    with open(output_filename, "w", encoding="utf-8") as out_file:
        out_file.write(result)

    print(f"\n✅ SUCCESS! Converted file saved as: {output_filename}")

if __name__ == "__main__":
    main()