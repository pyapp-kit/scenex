document$.subscribe(() => {
  const tableData = {
    Source: ["Github (dev version)"],
    Graphics: ["VisPy", "Pygfx"],
    Frontend: ["PyQt6", "wxPython", "Jupyter"],
  };

  const gitUrl = 'git+https://github.com/pyapp-kit/scenex.git@main';
  const commandMap = {
    "Github (dev version),VisPy,PyQt6": `pip install "scenex[vispy,pyqt6] @ ${gitUrl}"`,
    "Github (dev version),VisPy,wxPython": `pip install "scenex[vispy,wx] @ ${gitUrl}"`,
    "Github (dev version),VisPy,Jupyter": `pip install "scenex[vispy,jupyter] @ ${gitUrl}"`,
    "Github (dev version),Pygfx,PyQt6": `pip install "scenex[pygfx,pyqt6] @ ${gitUrl}"`,
    "Github (dev version),Pygfx,wxPython": `pip install "scenex[pygfx,wx] @ ${gitUrl}"`,
    "Github (dev version),Pygfx,Jupyter": `pip install "scenex[pygfx,jupyter] @ ${gitUrl}"`,
  };

  const container = document.getElementById("install-table");

  const createTable = () => {
    Object.keys(tableData).forEach((category) => {
      const label = document.createElement("div");
      label.classList.add("category-label");
      label.textContent = category;

      const buttonsContainer = document.createElement("div");
      buttonsContainer.classList.add("grid-right", "buttons");

      tableData[category].forEach((item, index) => {
        const button = document.createElement("button");
        button.textContent = item;
        button.dataset.value = item;

        // Activate the first button in the category
        if (index === 0) {
          button.classList.add("active");
        }

        // Event listener for button click
        button.addEventListener("click", (event) => {
          // Deactivate all buttons in this category
          Array.from(buttonsContainer.children).forEach((btn) => btn.classList.remove("active"));

          // Activate the clicked button
          button.classList.add("active");

          // Update command
          updateCommand();
        });

        buttonsContainer.appendChild(button);
      });

      container.appendChild(label);
      container.appendChild(buttonsContainer);
    });

    const label = document.createElement("div");
    label.classList.add("category-label", "command-label");
    label.textContent = "Command:";

    const commandDiv = document.createElement("div");
    commandDiv.classList.add("grid-right", "command-section");
    commandDiv.innerHTML = `
    <p id="command-output">Select options to generate command</p>
    <button class="md-clipboard md-icon" title="Copy to clipboard"></button>
    `;
    container.appendChild(label);
    container.appendChild(commandDiv);

    // Add copy functionality
    const copyButton = commandDiv.querySelector(".md-clipboard");
    copyButton.addEventListener("click", copyToClipboard);

    // Update the command output initially
    updateCommand();
  };

  const updateCommand = () => {
    const activeButtons = document.querySelectorAll("button.active");
    const selectedOptions = Array.from(activeButtons).map((btn) => btn.dataset.value);
    const commandOutput = document.getElementById("command-output");
    console.log();

    if (selectedOptions.length === 0) {
      commandOutput.textContent = "Select options to generate command";
    } else {
      commandOutput.textContent = commandMap[selectedOptions.join(",")];
    }
  };
  const copyToClipboard = () => {
    const commandOutput = document.getElementById("command-output").textContent;
    navigator.clipboard
      .writeText(commandOutput)
      .then(() => {
        // give a little animated feedback
        const commandDiv = document.querySelector(".command-section .md-clipboard");
        commandDiv.classList.add("copied");
        setTimeout(() => {
          commandDiv.classList.remove("copied");
        }, 500);
      })
      .catch((error) => {
        console.error("Failed to copy to clipboard", error);
      });
  };

  if (container) {
    createTable();
  }
});
