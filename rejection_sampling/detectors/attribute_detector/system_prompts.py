api_head_prompt1=f"Assess the two faces in the image for the following attributes:"
api_head_prompt2=f"Your output should only consist of a JSON that contains the attributes."
with_skin_tone_head_prompt="For all attributes except 'face_with_lighter_skin_tone', the output should be a list of two Yes/No responses."
wo_skin_tone_head_prompt="For all attributes the output should be a list of two Yes/No responses."
non_skin_prompt="The first Yes/No of the list should correspond to the left face and the second Yes/No should correspond to the right face." 
with_skin_prompt="For face_with_lighter_skin_tone, the output should be either 'Right face', 'No significant difference' or 'Left face'."

example_with_skin_tone="""
An example output would be:
{
    "<attribute1>":["Yes","No"],
    "<attribute2>":["Yes","Yes"],
    ...
    "face_with_lighter_skin_tone":"Right face"
}
"""

example_without_skin_tone="""
An example output would be:
{
    "<attribute1>":["Yes","No"],
    "<attribute2>":["Yes","Yes"],
    ...
}
"""

def make_prompt_based_on_attributes(attributes_list):
    final_prompt=api_head_prompt1 +"\n"
    for attr in attributes_list:
        if attr not in ['light_colored_skin_tone','dark_colored_skin_tone']:
            final_prompt+=f"{attr}\n"

    final_prompt+=f"face_with_lighter_skin_tone\n"
    # if 'light_colored_skin_tone' in attributes_list or 'dark_colored_skin_tone' in attributes_list:
    #     final_prompt+=f"face_with_lighter_skin_tone\n"
    
    final_prompt+=api_head_prompt2

    #if 'light_colored_skin_tone' in attributes_list or 'dark_colored_skin_tone' in attributes_list:
    final_prompt+=with_skin_tone_head_prompt +"\n"
    final_prompt+=non_skin_prompt + "\n"
    final_prompt+=with_skin_prompt + "\n"
    final_prompt+=example_with_skin_tone +"\n"
    # else:
    #     final_prompt+=wo_skin_tone_head_prompt +"\n"
    #     final_prompt+=non_skin_prompt + "\n"
    #     final_prompt+=example_without_skin_tone +"\n"
    
    return final_prompt.strip()
