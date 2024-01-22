const video = document.getElementById('video');
const fileInput = document.getElementById('fileInput');

let videoStream;
let frameCount = 0;

fileInput.addEventListener('change', (event) => {
  const file = event.target.files[0];

  if (file) {
    console.log(file);
    const videoURL = URL.createObjectURL(file);
    video.src = videoURL;

    video.style.display = 'block';
    console.log('Selected Video File:', file.path || file.name);

     document.cookie = `fileName=${file.name}`;

    // Submit the form
//    this.submit();
  }

});

function startProcessing() {
  displayImage();
  if (!video.src) {
    console.error('No video source selected.');
    return;
  }

  Promise.all([
    faceapi.nets.tinyFaceDetector.loadFromUri('static/models'),
    faceapi.nets.faceLandmark68Net.loadFromUri('static/models'),
    faceapi.nets.faceRecognitionNet.loadFromUri('static/models'),
    faceapi.nets.faceExpressionNet.loadFromUri('static/models')
  ]).then(processVideo);
}

function processVideo() {
  video.play();

  const processFrames = async () => {
    if (video.videoWidth && video.videoHeight) {
      const displaySize = { width: video.videoWidth, height: video.videoHeight };

      const detections = await faceapi.detectAllFaces(video, new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks().withFaceExpressions();
      const resizedDetections = faceapi.resizeResults(detections, displaySize);

      if (detections.length > 0) {
        captureFaceImage(resizedDetections, frameCount);
      }

      frameCount++;
    }
  };

  const checkVideoEnded = () => {
    if (video.ended) {
      console.log('Video ended. Script stopped.');
      const selectedImages = ['frame_0.png', 'frame_3.png', 'frame_5.png', 'frame_7.png', 'frame_9.png','frame_2.png'];

      clearInterval(frameIntervalId);
      const imagesString = selectedImages.join(',');
      document.cookie = `Images=${imagesString}`;
      displaySelectedImages(selectedImages);





    }
  };

  const frameIntervalId = setInterval(() => {
    processFrames();
    checkVideoEnded();
  }, 100); // Adjust the interval as needed
}



function captureFaceImage(detections, frameNumber) {
  // Assuming you want to capture the first face detected
  const face = detections[0];

  // Create a canvas to extract the face image
  const faceCanvas = document.createElement('canvas');
  const faceCanvasContext = faceCanvas.getContext('2d');
  faceCanvas.width = face.detection.box.width;
  faceCanvas.height = face.detection.box.height;

  // Draw the face on the new canvas
  faceCanvasContext.drawImage(video, face.detection.box.x, face.detection.box.y, face.detection.box.width, face.detection.box.height, 0, 0, face.detection.box.width, face.detection.box.height);

  // Convert the canvas content to a data URL
  const dataURL = faceCanvas.toDataURL('image/png');

  // Save the data URL to the local 'frames' directory
  const downloadLink = document.createElement('a');
  downloadLink.href = dataURL;
  downloadLink.download = `frame_${frameNumber}.png`;
  downloadLink.click();
}

function displaySelectedImages(images) {
    const selectedImagesContainer = document.getElementById('selectedImagesContainer');

    selectedImagesContainer.style.zIndex = '999';
    selectedImagesContainer.style.backgroundColor = 'black';

    selectedImagesContainer.innerHTML = ''; // Clear previous content

    // Create the grid container
    const gridContainer = document.createElement('div');
    gridContainer.classList.add('grid-container');

    // Set the desired width for the grid container (e.g., 70%)
    gridContainer.style.width = '100%';


    // Display the selected images in the grid
    images.forEach(image => {
        const gridItem = document.createElement('div');

        gridItem.classList.add('grid-item');
             gridItem.style.marginTop = '2px';


        const imgElement = document.createElement('img');
        imgElement.src = `static/Frames/${image}`;
        imgElement.alt = 'frame';
        imgElement.classList.add('img-fluid');

        // Set the desired size for the images (e.g., 100% width, 100% height)
        imgElement.style.width = '95%';
        imgElement.style.height = '95%';

        // Append the image to the grid item
        gridItem.appendChild(imgElement);

        // Append the grid item to the grid container
        gridContainer.appendChild(gridItem);
    });

    // Append the grid container to the selectedImagesContainer
    selectedImagesContainer.appendChild(gridContainer);
}

function displayImage() {
            imageSrc = 'static/Images/loader.gif';
            const imageContainer = document.getElementById('xyz');

            // Create an img element
            const imgElement = document.createElement('img');
            imgElement.src = imageSrc;
            imgElement.alt = 'Image';
            imgElement.style.width = '50%';
            imgElement.style.position = 'absolute';
            imgElement.style.top = '15%';
            imgElement.style.Height = '50%';
            imgElement.style.marginLeft = '160px';

            // Append the img element to the imageContainer
            imageContainer.appendChild(imgElement);

    }

 document.getElementById('fileInput').addEventListener('change', function () {
        var label = document.querySelector('.label');
        var fileInput = document.getElementById('fileInput');
        var fileLabel = document.querySelector('.label');

        if (fileInput.files.length > 0) {
            fileLabel.textContent = fileInput.files[0].name;
            document.querySelector('.file-input').classList.add('-chosen');

            var fileName = fileInput.files[0].name;
            fileLabel.textContent = fileName;

            var cookies = document.cookie.split(';');
            document.cookie = `Images=${fileName}`;

        } else {
            fileLabel.textContent = 'No file selected';
            document.querySelector('.file-input').classList.remove('-chosen');
        }
    });

