# Cognitive Feedback System Simulator (CFSS)

## Summary

A dynamic, adjustable simulator modeling feedback loops among an agent's mind, body, and environment. The system includes modifiable stressors, regulatory strategies, and environmental factors. The goal is to explore how internal and external conditions interact through cognitive and physical regulation strategies—leading to either breakdown or resilience.

## Main Goal
Explore how mind-body-environment feedback loops can be simulated and manipulated for insight or control.

## Core Components

1. Agent Initialization

Create a central agent (user-defined or default)

Parameters:
	- Internal: Cognitive load, neurochemical balance, bodily stress, pain, control need
	- External: Temperature, noise, confinement, social contact
	- Regulatory capacities: Breathing techniques, cognitive reframing, pharmacological intervention, nutritional state, sleep status

2. Modular Definitions

	- Mind: Formal, functional entity (nonphysical, but instantiated in a system; could be simulated or abstracted)
	- Body: The substrate of internal regulation. Define scope per simulation: CNS only, Full nervous system, Organs, or Tool-extended self?
	- Environment: External conditions relative to the chosen "body level". 

Thus: Add a "Boundary Scope Toggle": change the level of what's considered self vs world (e.g., brain-only vs full-organism vs body+cane)

3. Nutrition Module

Add as a core interactive parameter

Possible subfactors:
	- Macronutrient balance (simple carbs vs complex)
	- Specific neurotransmitter precursors (e.g., tryptophan, tyrosine)
	- Gut-brain connection (brief model or placeholder for future expansion)
	- Deficiency effects over time (simulate B12, magnesium, etc.)

4. System Dynamics

Define feedback loops:
	- Stress → cognitive load → regulatory strategy used → outcome → updated internal state
	- Poor nutrition → lowered serotonin/dopamine → weaker regulation → breakdown cycle

Add a stochasticity layer (probabilistic element to simulate chaos/noise)

5. User Interaction & Visualization
Interface: Clean + engineer-professional, with subtle retro touches


## Initial Features:
	Real-time or tick-based simulation
	Sliders or input fields for parameters

## Visual output:
	Graph of variables over time
	System map view (simple node-arrow SVG or grid layout)

Optional: A toggle to switch visual perspective (e.g., zoom from brain to organism to environment)

## Constraints
- Platform: Windows 11
- No HTML for initial version (Python only; console or simple GUI like tkinter, pygame, or matplotlib for visuals)

Intended First Version Completion: May 17

Later possibility: Relax HTML restriction to allow richer visuals (SVG animations, etc.)

## Short-Term Milestones

- Base agent & parameters	1 hr	

- Set up rules/feedback logic	2 hrs	

- Simple data visualizer (matplotlib or pygame)	2–3 hrs	

- Add nutrition module with placeholder logic	1 hr	

- Build environment-body-mind boundary selector	1 hr

- Add references & start MD doc	1 hr	

- Lit search for key values (e.g., cortisol response curves, tryptophan→serotonin pathway)	2 hrs (ongoing)	

## References & Realism

Add .md or .pdf with basic scientific citations:

	- Neurotransmitter production from diet
	- Stress hormone response dynamics
	- Thermoregulation and cognitive clarity
	- Regulation strategies in psychophysiology
	- Allow direct clicking from the UI to these references later via doc links or tooltips.

## Design Goals

Aesthetic: Clean + slightly nostalgic

	Grayscale/blue wireframe-style overlays

	Subtle pixel fonts or button styles where fitting

	Functionality before polish, but UI should be intuitive from the start

Later polish: Add SVG or animated schematic (in JS or Python+tk/svg toolkit)
