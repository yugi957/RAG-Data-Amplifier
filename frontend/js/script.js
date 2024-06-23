const socket = io('http://127.0.0.1:5000');

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
    const tagBox = document.querySelector('.tag-box');
    tagBox.innerHTML = '';
    tags.forEach(tag => {
        const button = document.createElement('button');
        button.className = 'tag';
        button.textContent = tag;
        button.onclick = () => button.classList.toggle('selected');
        tagBox.appendChild(button);
    });
}

async function augmentData() {
    const augmentDataBtn = document.getElementById('augmentDataBtn');
    augmentDataBtn.disabled = true;
    augmentDataBtn.textContent = "Processing...";

    const selectedTags = Array.from(document.querySelectorAll('.tag.selected')).map(button => button.textContent);
    const modifier = document.getElementById('modifierInput').value;

    try {
        const response = await fetch('http://127.0.0.1:5000/augment', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ tags: selectedTags, modifier: modifier })
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

async function generateClusters() {
    // Implement the logic to generate clusters here
    alert("Generate Clusters button clicked!");
}
