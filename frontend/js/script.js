const socket = io('http://127.0.0.1:5000');
let queries = {};

socket.on('progress', (data) => {
    const progressBar = document.getElementById('progressBar');
    progressBar.style.width = data.progress + '%';
    if (data.progress === 100) {
        document.getElementById('augmentDataBtn').disabled = false;
    }
});

socket.on('completed', async (data) => {
    await fetchTags();
});

socket.on('augment_progress', (data) => {
    const augmentProgressBar = document.getElementById('augmentProgressBar');
    augmentProgressBar.style.width = data.progress + '%';
});

socket.on('augment_completed', (data) => {
    displaySampleTexts(data.sample_texts);
    displayDataAnalysis(data.charts);
    document.getElementById('downloadGeneratedData').disabled = false;
    confetti({
        particleCount: 100,
        spread: 70,
        origin: { y: 0.6 }
    });
});

function showFileName() {
    const fileInput = document.getElementById('fileUpload');
    const fileLabel = document.getElementById('fileLabel');
    const fileName = fileInput.files[0].name;
    fileLabel.textContent = fileName;
}

async function sendData() {
    const fileInput = document.getElementById('fileUpload');
    const sendDataBtn = document.getElementById('sendDataBtn');
    const progressBar = document.getElementById('progressBar');
    const augmentDataBtn = document.getElementById('augmentDataBtn');

    if (fileInput.files.length === 0) {
        alert("Please select a file first.");
        return;
    }

    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('file', file);

    sendDataBtn.disabled = true;
    sendDataBtn.textContent = "Sending...";

    try {
        const response = await fetch('http://127.0.0.1:5000/upload', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Network response was not ok ' + response.statusText);
        }

        const data = await response.json();
        console.log('Success:', data);
        sendDataBtn.textContent = "Sent!";
        progressBar.style.width = '0%';

    } catch (error) {
        console.error('Error:', error);
        alert('Error sending data: ' + error.message);
        sendDataBtn.disabled = false;
        sendDataBtn.textContent = "Send Data";
    }
}

async function fetchTags() {
    try {
        const response = await fetch('http://127.0.0.1:5000/tags');
        if (!response.ok) {
            throw new Error('Network response was not ok ' + response.statusText);
        }
        const data = await response.json();
        populateTags(data.tags);
    } catch (error) {
        console.error('Error:', error);
        alert('Error fetching tags: ' + error.message);
    }
}

function populateTags(tags) {
    console.log('Tags:', tags);
    const tagBox = document.querySelector('.tag-box');
    tagBox.innerHTML = '';
    Object.keys(tags).forEach(tag => {
        const button = document.createElement('button');
        button.className = 'tag';
        button.textContent = tag;
        button.onclick = () => toggleDropdown(tags[tag], button);
        tagBox.appendChild(button);
    });
}

function toggleDropdown(data, button) {
    console.log(data)
    button.classList.toggle('selected');
    const dropdownContainer = document.querySelector('.dropdown-container');
    dropdownContainer.innerHTML = '';  // Clear previous dropdowns
    if (button.classList.contains('selected')) {
        const dropdownMenu = document.createElement('div');
        dropdownMenu.className = 'dropdown-menu active';
        // Replace these items with actual data from your server
        if (data[0] === 'object') {
            const items = data[1];
            queries[button.textContent] = [data[0], [], null];
            items.forEach(item => {
                const dropdownItem = document.createElement('div');
                dropdownItem.className = 'dropdown-item';
                dropdownItem.textContent = item;
                dropdownItem.onclick = () => {
                    dropdownItem.classList.toggle('selected');
                    toggleSelectedItem(button.textContent, dropdownItem.textContent);
                };
                dropdownMenu.appendChild(dropdownItem);
            });
            dropdownContainer.appendChild(dropdownMenu);
        }
        else {
            const textEntryDiv = document.createElement('div');
            textEntryDiv.className = 'text-entry';

            const input1 = document.createElement('input');
            input1.type = 'text';
            input1.placeholder = '0';
            input1.className = 'text-input';

            const input2 = document.createElement('input');
            input2.type = 'text';
            input2.placeholder = '0';
            input2.className = '0';

            const separator = document.createTextNode(' - ');

            textEntryDiv.appendChild(input1);
            textEntryDiv.appendChild(separator);
            textEntryDiv.appendChild(input2);

            queries[button.textContent] = [data[0], null, null];

            input1.addEventListener('input', () => updateSelectedEntries(button.textContent, input1.value, input2.value));
            input2.addEventListener('input', () => updateSelectedEntries(button.textContent, input1.value, input2.value));

            dropdownContainer.appendChild(textEntryDiv);
        }
    }
    else {
        delete queries[button.textContent];
    }
}

function updateSelectedEntries(tag, value1, value2) {
    queries[tag][1] = parseFloat(value1);
    queries[tag][2] = parseFloat(value2);
}

function toggleSelectedItem(tag, subtag) {
    const index = queries[tag][1].indexOf(subtag);
    if (index === -1) {
        queries[tag][1].push(subtag);
    } else {
        queries[tag][1].splice(index, 1);
    }
    console.log('Selected items:', queries);
}

async function augmentData() {
    const augmentDataBtn = document.getElementById('augmentDataBtn');
    console.log('Augmenting Data:', queries);
    augmentDataBtn.disabled = true;
    augmentDataBtn.textContent = "Processing...";

    const selectedTags = Array.from(document.querySelectorAll('.tag.selected')).map(button => button.textContent);
    const modifier = document.getElementById('modifierInput').value;
    console.log('JSON Output:', { tags: selectedTags, items: queries, modifier: modifier });
    try {
        const response = await fetch('http://127.0.0.1:5000/augment', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ tags: selectedTags, items: queries, modifier: modifier })
        });

        if (!response.ok) {
            throw new Error('Network response was not ok ' + response.statusText);
        }

        const data = await response.json();
        console.log('Augment Success:', data);
        augmentDataBtn.disabled = false;
        augmentDataBtn.textContent = "Augment Data";

    } catch (error) {
        console.error('Error:', error);
        alert('Error augmenting data: ' + error.message);
        augmentDataBtn.disabled = false;
        augmentDataBtn.textContent = "Augment Data";
    }
}

function displaySampleTexts(sampleTexts) {
    const sampleTextDiv = document.querySelector('.sample-text');
    sampleTextDiv.innerHTML = '';
    sampleTexts.forEach(text => {
        const p = document.createElement('p');
        p.textContent = text;
        sampleTextDiv.appendChild(p);
    });
}

function displayDataAnalysis(charts) {
    const analysisDiv = document.querySelector('.analysis');
    analysisDiv.innerHTML = '';
    charts.forEach(chart => {
        const canvas = document.createElement('canvas');
        analysisDiv.appendChild(canvas);
        new Chart(canvas, chart);
    });
}

async function displayDataAnalysis() {

}

async function downloadFile() {
    try {
        const response = await fetch("http://127.0.0.1:5000/download", {
            method: 'GET',
            headers: {
                'Content-Type': 'application/octet-stream'
            }
        });

        if (!response.ok) {
            throw new Error('Network response was not ok ' + response.statusText);
        }

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = 'output.csv'; // Set the filename dynamically if needed
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);

        // Trigger confetti effect
        const button = document.getElementById('downloadGeneratedData');
        const rect = button.getBoundingClientRect();
        confetti({
            particleCount: 100,
            spread: 70,
            origin: {
                x: (rect.left + rect.width / 2) / window.innerWidth,
                y: (rect.top + rect.height / 2) / window.innerHeight
            }
        });
    } catch (error) {
        console.error('Error downloading file:', error);
        alert('Error downloading file: ' + error.message);
    }
}
