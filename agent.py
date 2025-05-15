class Agent:
    def __init__(self, name="DefaultAgent"):
        # Give the agent a name
        self.name = name

        # What's going on inside the agent
        self.internal_state = {
            "pain": 0.2,
            "instability": 0.4,
            "need_for_control": 0.5
        }
        #  todo: Add more, parameterize 

        # External surroundings affecting the agent
        self.environment = {
            "temperature": 22.0,
            "confinement": 0.2,
            "social_contact": 0.5
        }

        # Methods the agent can use to regulate themselves
        self.regulation = {
            "breathing": 0.5,
            "cognitive_override": 0.4,
            "pharmacology": 0.0
        }

        # Nutritional factors that affect mood and regulation
        self.nutrition = {
            "glucose_level": 0.8,
            "tryptophan": 0.5,
            "hydration": 0.9
        }

        # History of previous states (you can fill this in later)
        self.history = []

    #Method that defines what print(agent) will show
    def __str__(self):
        return f"<Agent {self.name}>"


# Only runs if this file is executed directly (not imported elsewhere)
if __name__ == "__main__":
    agent = Agent(name="Athena")
    print(agent)
    print("Internal State:", agent.internal_state)
    print("Environment:", agent.environment)

#keep window open if opening from windows explorer
    input("\nPress Enter to exit...")

