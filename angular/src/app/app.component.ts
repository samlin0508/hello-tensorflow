import { Component, ElementRef, OnInit, ViewChild } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css'
})
export class AppComponent implements OnInit {
  title = 'angular';
  canvasWidth = 640;
  canvasHeight = 480;

  @ViewChild('videoCamera', {static: true}) 
  videoCamera: ElementRef | undefined;

  @ViewChild('canvas', {static: true}) 
  canvas: ElementRef | undefined;

  ngOnInit(): void {
    this.initWebCam();
    // this.buildModel();
  }

  async initWebCam() {
    navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: 'environment'
      }
    }).then(stream => {
      if(this.videoCamera) {
        this.videoCamera.nativeElement.srcObject = stream;
        this.videoCamera.nativeElement.onloadedmetadata = () => {
          if(this.videoCamera) {
            this.videoCamera.nativeElement.play();
            // this.predictImageClassification();
            cocoSsd.load().then(model => {
              setInterval(() => {
                model.detect(this.videoCamera?.nativeElement).then(predictions => {
                  console.log('Predictions: ', predictions);

                  let ctx = this.canvas?.nativeElement.getContext("2d");
                  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
                  ctx.drawImage(
                    this.videoCamera?.nativeElement,
                    0,
                    0,
                    this.canvasWidth,
                    this.canvasHeight
                  );
    
                  predictions.forEach(prediction => {
                    ctx.strokeStyle = '#0074df';
                    ctx.lineWidth = 2;
                    ctx.strokeRect(
                      prediction.bbox[0], 
                      prediction.bbox[1], 
                      prediction.bbox[2], 
                      prediction.bbox[3]
                    );

                    ctx.strokeStyle = '#ff0000';
                    ctx.font = '16px open-sans';
                    ctx.textBaseline = 'top';
                    ctx.strokeText(
                      prediction.class,
                      prediction.bbox[0]+3, 
                      prediction.bbox[1], 
                    )
                  });
                });
              }, 100);
            });
          }
        };
      }
    });
  }

  // async analyzeSentiment(text: string) {
  //   const model = await tf.loadLayersModel(
  //     "https://path/to/your/sentiment-analysis/model.json"
  //   );
  //   const tokens = tokenize(text); // Implement a tokenize function to tokenize the input text
  //   const input = tf.tensor1d(tokens).expandDims(0);
  
  //   const output = model.predict(input) as tf.Tensor;
  //   const sentiment = output.dataSync()[0];
  
  //   console.log("Sentiment score:", sentiment);
  // }
  
  async predictImageClassification() {
    const model = await tf.loadLayersModel(
      "https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json"
    );
    const input = tf.browser
      .fromPixels(this.videoCamera?.nativeElement)
      .resizeBilinear([224,224])
      .toFloat()
      .expandDims();
    const normalized = input.div(255);
    const predictions = model.predict(normalized) as tf.Tensor;

    const topK = 5;
    const topKIndices = tf.topk(predictions, topK).indices.dataSync();
  
    console.log("Top", topK, "predictions:");
    for (let i = 0; i < topKIndices.length; i++) {
      console.log(`#${i + 1}: ${topKIndices[i]}`);
    }
  }

  buildModel() {
    // Define a model for linear regression.
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

    // Prepare the model for training: Specify the loss and the optimizer.
    model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });

    // Generate some synthetic data for training.
    const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
    const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);

    // Train the model using the data.
    model.fit(xs, ys).then(() => {
      // Use the model to do inference on a data point the model hasn't seen before:
      (model.predict(tf.tensor2d([5], [1, 1])) as tf.Tensor).print();
    });
  }
}
