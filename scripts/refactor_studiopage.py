import re

with open("../claudio/frontend/src/pages/StudioPage.tsx", "r") as f:
    text = f.read()

# Add import
text = text.replace("import WaveformViewer from '../components/WaveformViewer';", "import WaveformViewer from '../components/WaveformViewer';\nimport StudioEffectsChain from '../components/StudioEffectsChain';")

# Remove state variables (lines 40 to 64)
# We can do this safely via regex
state_vars = r"  // Effects\s+const \[reverb, setReverb\].*?setEqHigh\(0\);\n"
text = re.sub(state_vars, "", text, flags=re.DOTALL)

# Remove useEffect block (lines 99 to 120)
useeffects = r"  // Sync effects → engine.*?}?, \[eqHigh\]\);\n"
text = re.sub(useeffects, "", text, flags=re.DOTALL)

# Remove JSX block (lines 521 to 650)
# We will use string searching for boundaries.
start_jsx = r"          {/\* Effects row \*/}"
end_jsx = r"          {/\* Audio File Player \*/}"
text = re.sub(f"{start_jsx}.*?(?={end_jsx})", "          <StudioEffectsChain engine={engineRef.current} ready={ready} grReduction={grReduction} />\n\n", text, flags=re.DOTALL)

with open("../claudio/frontend/src/pages/StudioPage.tsx", "w") as f:
    f.write(text)
