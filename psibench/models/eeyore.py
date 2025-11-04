from string import Template

system_prompt_template = '''You will act as a help-seeker struggling with negative emotions in a conversation with someone who is listening to you.
YOUR PROFILE:
${name}${gender}${age}${marital_status}${occupation}${situation_of_the_client}${counseling_history}${resistance_toward_the_support}${symptom_severity}${cognition_distortion_exhibition}${depression_severity}${suicidal_ideation_severity}${homicidal_ideation_severity}
YOUR TASK:
As the client, your role is to continue the conversation by responding naturally to the supporter, reflecting the characteristics outlined in your profile.'''

def validate_input(input):
    if input is None:
        return ""
    if input == "" or input.lower() == "not specified" or input.lower() == "unknown" or input.lower()=="n/a" or "cannot be identified" in input.lower() or "cannot be determined" in input.lower() or "not mention" in input.lower() or "not exhibited" in input.lower():
        return ""
    else:
        return input

def prepare_prompt_from_profile(prof=None):
    name_tb = prof.get("name", "").lower()
    age_tb = prof.get("age", "").lower()
    gender_dd = prof.get("gender", "").lower()
    occp_tb = prof.get("occupation", "").lower()
    marital_dd = prof.get("marital status", "").lower()
    sit_tb = prof.get("situation of the client", "").lower()
    history_tb = prof.get("counseling history", "")
    resis_cb = prof.get("resistance toward the support", "").lower()

    mild_sym_dd = [k.lower() for k, v in prof.get("symptom severity", {}).items() if "mild" in v.lower()]
    mod_sym_dd = [k.lower() for k, v in prof.get("symptom severity", {}).items() if "moderate" in v.lower()]
    seve_sym_dd = [k.lower() for k, v in prof.get("symptom severity", {}).items() if "severe" in v.lower()]

    mild_cog_dd = [k.lower() for k, v in prof.get("cognition distortion exhibition", {}).items() if "not exhibited" not in v.lower()]
    mod_cog_dd = []
    seve_cog_dd = []

    overall_dd = prof.get("depression severity", "")
    suicidal_dd = prof.get("suicidal ideation severity", "")
    homicidal_dd = prof.get("homicidal ideation severity", "")


    return get_system_prompt_with_profile(name_tb, age_tb, gender_dd, occp_tb, marital_dd, sit_tb, history_tb,resis_cb, mild_sym_dd, mod_sym_dd, seve_sym_dd, mild_cog_dd, mod_cog_dd, seve_cog_dd, overall_dd, suicidal_dd, homicidal_dd)

def get_system_prompt_with_profile(name_tb, age_tb, gender_dd, occp_tb, marital_dd, sit_tb, history_tb, resis_cb, mild_sym_dd, mod_sym_dd, seve_sym_dd, mild_cog_dd, mod_cog_dd, seve_cog_dd, overall_dd, suicidal_dd, homicidal_dd):
    """
    This function gets the system prompt with the profile dictionary.
    """

    profile_dict = {"name":"", "gender":"", "age":"", "marital_status":"", "occupation":"", "situation_of_the_client":"", "counseling_history":"", "resistance_toward_the_support":"", "symptom_severity":"", "cognition_distortion_exhibition":"", "depression_severity":"", "suicidal_ideation_severity":"", "homicidal_ideation_severity":""}
    system_prompt = parse_system_prompt(name_tb, age_tb, gender_dd, occp_tb, marital_dd, sit_tb, history_tb, resis_cb, mild_sym_dd, mod_sym_dd, seve_sym_dd, mild_cog_dd, mod_cog_dd, seve_cog_dd, overall_dd, suicidal_dd, homicidal_dd)
    patient_profile = "## PROFILE\n" + system_prompt.split("YOUR PROFILE:")[-1].split("YOUR TASK:")[0]

    profile_dict["name"] = validate_input(name_tb)
    profile_dict["age"] = validate_input(age_tb)
    profile_dict["gender"] = validate_input(gender_dd)
    profile_dict["occupation"] = validate_input(occp_tb)
    profile_dict["situation_of_the_client"] = validate_input(sit_tb)
    profile_dict["marital_status"] = validate_input(marital_dd)
    profile_dict["resistance_toward_the_support"] = validate_input(resis_cb)
    profile_dict["counseling_history"] = validate_input(history_tb)
    profile_dict["symptom_severity_mild"] = mild_sym_dd
    profile_dict["symptom_severity_moderate"] = mod_sym_dd
    profile_dict["symptom_severity_severe"] = seve_sym_dd
    profile_dict["cognitive_distortion"] = mild_cog_dd
    profile_dict["depression_severity"] = overall_dd
    profile_dict["suicidal_ideation_severity"] = validate_input(suicidal_dd)
    profile_dict["homicidal_ideation_severity"] = validate_input(homicidal_dd)

    return system_prompt, patient_profile, profile_dict

def parse_system_prompt(name_tb, age_tb, gender_dd, occp_tb, marital_dd, sit_tb, history_tb,resis_cb, mild_sym_dd, mod_sym_dd, seve_sym_dd, mild_cog_dd, mod_cog_dd, seve_cog_dd, overall_dd, suicidal_dd, homicidal_dd):
    temp_profile_dict = {"name":"", "gender":"", "age":"", "marital_status":"", "occupation":"", "situation_of_the_client":"", "counseling_history":"", "resistance_toward_the_support":"", "symptom_severity":"", "cognitive_distortion":"", "depression_severity":"", "suicidal_ideation_severity":"", "homicidal_ideation_severity":""}
    if validate_input(name_tb):
        temp_profile_dict["name"] = "- " + "name" + ": " + validate_input(name_tb) + "\n"
    if validate_input(age_tb):
        temp_profile_dict["age"] = "- " + "age" + ": " + validate_input(age_tb) + "\n"
    if validate_input(gender_dd):
        temp_profile_dict["gender"] = "- " + "gender" + ": " + validate_input(gender_dd) + "\n"
    if validate_input(occp_tb):
        temp_profile_dict["occupation"] = "- " + "occupation" + ": " + validate_input(occp_tb) + "\n"
    if validate_input(sit_tb):
        temp_profile_dict["situation_of_the_client"] = "- " + "situation of the client" + ": " + validate_input(sit_tb) + "\n"
    if validate_input(marital_dd):
        temp_profile_dict["marital_status"] = "- " + "marital status" + ": " + validate_input(marital_dd) + "\n"
    if validate_input(resis_cb):
        temp_profile_dict["resistance_toward_the_support"] = "- " + "resistance toward the support" + ": " + validate_input(resis_cb) + "\n"
    if validate_input(history_tb):
        temp_profile_dict["counseling_history"] = "- " + "counseling history" + ": " + validate_input(history_tb) + "\n"

    sup = ""
    for item in seve_sym_dd:
        sup += "  - " + str(item) + ": " + "severe" + "\n"
    for item in mod_sym_dd:
        sup += "  - " + str(item) + ": " + "moderate" + "\n"
    for item in mild_sym_dd:
        sup += "  - " + str(item) + ": " + "mild" + "\n"
    if sup:
        temp_profile_dict["symptom_severity"] = "- " + "symptom severity" + "\n" + sup


    sup = ""
    for item in seve_cog_dd:
        sup += "  - " + str(item) + ": " + "severe" + "\n"
    for item in mod_cog_dd:
        sup += "  - " + str(item) + ": " + "moderate" + "\n"
    for item in mild_cog_dd:
        sup += "  - " + str(item) + ": " + "exhibited" + "\n"
    if sup:
        temp_profile_dict["cognition_distortion_exhibition"] = "- " + "cognition distortion exhibition" + "\n" + sup

    if validate_input(overall_dd):
        temp_profile_dict["depression_severity"] = "- " + "depression severity" + ": " + validate_input(overall_dd) + "\n"

    if validate_input(suicidal_dd):
        temp_profile_dict["suicidal_ideation_severity"] = "- " + "suicidal ideation severity" + ": " + validate_input(suicidal_dd) + "\n"

    if validate_input(homicidal_dd):
        temp_profile_dict["homicidal_ideation_severity"] = "- " + "homicidal ideation severity" + ": " + validate_input(homicidal_dd) + "\n"

    system_prompt = Template(system_prompt_template).safe_substitute(temp_profile_dict)

    return system_prompt