import * as tf from '@tensorflow/tfjs-node';
import { readFileSync, readdirSync } from 'fs';

const loadModel = async () => {
    return await tf.loadLayersModel('file://src/data/model/model.json');
};

const getNormalizedImages = () => {
    const imageFolder = 'src/data/images';

    const images = readdirSync(imageFolder).map((imageFile) => {
        const imagePath = `${imageFolder}/${imageFile}`;
        const imageBuffer = readFileSync(imagePath);

        const decodedImage = tf.node.decodeImage(imageBuffer);
        const normalizedImage = decodedImage.div(255.0);

        const resizedImage = tf.image.resizeBilinear(normalizedImage, [28, 28]);

        const reshapedImage = resizedImage.reshape([1, 28, 28, 1]);

        return reshapedImage;
    });

    return images;
};

const model = await loadModel();
const images = getNormalizedImages();

const results = new Array(10).fill(0);

for (let i = 0; i < images.length; i++) {
    const prediction = model.predict(images[i]);
    const digit = prediction.argMax(1).dataSync()[0];

    results[digit]++;
}

console.log(results);
