

class imageClassifier {

    
    constructor(numClasses) {

        //Webcam 
        this.webcam = null; 

        //Variables to hold models 
        this.mobilenet = null; 
        this.model = null; 
        //Number of classes 
        this.numClasses = numClasses;
        this.layerSize =  224;
        //Variables to store data and labels for predictions
        this.label = []; 
        this.dataset = null; 
        this.img = null; 
    }

    async initalise() {
        //Dataset object which will store the activations 
        this.dataset = new dataset(this.numClasses); 

        //Load Model
        const net = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');

        //Warming up the model
        net.predict(tf.zeros([1, 224, 224, 3])).dispose(); 

        //Layer we will be using as a feature extractor 
        const layer = net.getLayer('conv_pw_13_relu');
        //Creating model based on inputs

        this.mobilenet = tf.model({inputs: net.input, outputs: layer.output});
        console.log(this.mobilenet);
        for(var a = 0; a < 82; a++ ) {
            const layer = this.mobilenet.getLayer(null, a)
            // if(a > 63) { 
            // layer.trainable = true; 
            // } else {
                layer.trainable = false;
            // }
            console.log(layer.trainable)
        }
        
        this.mobilenet.summary();
        console.log("Model Has Been Loaded ")
        return true; 
     
    }


    async addLabel(label) {
        this.img = await this.getTensor();
        this.dataset.addSample(this.mobilenet.predict(this.img), label); 

        this.label = this.dataset.getLabel(); 
    }




    

    async addLabelFromUpload(img, label) {
        this.img = await this.getTensorFromUpload(img);
        this.dataset.addSample(this.mobilenet.predict(this.img), label); 

        this.label = this.dataset.getLabel(); 
    }

    async getTensorFromUpload(img) {
        const tImg =  tf.browser.fromPixels(img); 

        let tensor = tf.image.resizeBilinear(tImg, [224, 224]); 
       
        const processedImage = tf.tidy(() => tensor.expandDims(0).toFloat().div(tf.scalar(127)).sub(tf.scalar(1)));
        tensor.dispose();

        return processedImage;  
    }



        async createWebcam() {

             const webcamElement =  await document.getElementById('webcam');
             this.webcam = await tf.data.webcam(webcamElement);

            console.log('Webcam Loaded')
            //To capture an image 
        }



        /**
         * Provides predictions for a singular image. 
         * Method conducts a sanity check to see if the model is trained 
         * 
         */
        async predict () {
            if(!this.mobilenet) {
                console.log('MobileNet not loaded');
            } else if (this.mobilenet == null) {
                console.log('Model not trained')
            }

                this.img = await this.getTensor()
        
                await tf.nextFrame(); 

                //Determining the predictions for each image 

                const classPredictions = tf.tidy(() => {
                const imgTensor = this.img; 
                const imageFeatures = this.mobilenet.predict(imgTensor);
                const predictions = this.model.predict(imageFeatures);
                return predictions.as1D().argMax(); 

            });

            

            const classId = (await classPredictions.data())[0]; 
            //Returning class label 
            console.log("Label " + this.label[classId]); 

            return this.label[classId]; 
            
        }



        /**
         * takes a frame from webcam and normalises it 
         * Add batch element onto tensor
         * 
         */
         async getTensor() {
            const img =  await this.webcam.capture();
           
            const processedImage = tf.tidy(() => img.expandDims(0).toFloat().div(tf.scalar(127)).sub(tf.scalar(1)));
            img.dispose();

            return processedImage;  
        }



        async trainModel(callback, hiddenUnits, learningRate, trainingEpochs) {
            


            if(hiddenUnits == null && learningRate == null && trainingEpochs == null )
            {
            hiddenUnits = 300; 
            learningRate = 0.0001;
            trainingEpochs = 50;             
            }

            let labelSize = this.dataset.getLabelLength();

            if(!this.mobilenet) {
                throw new Error ("MobileNet not loaded")
            } else if (this.labelSize < this.numClasses) {
                throw new Error ("Insufficient training data")
            }

            //Optimiser to assist with training 
            const optimiser = tf.train.adam(learningRate);

            const l1 = tf.regularizers.l1();
            //Building a model which takes feature tensors as input and then predicts classes as outputs
            // Last
            this.model = tf.sequential({
                layers: [
            //Flatterns the input tensor so it can be used for the dense layer. No training
            //Occurs here and instead we focus on preforming a reshape
                    tf.layers.flatten(
                        {inputShape: this.mobilenet.outputs[0].shape.slice(1)}),
            //Layer 1
                    tf.layers.dense({
                        units: hiddenUnits,
                        activation: 'relu',
                        useBias: true
                    }),
                    tf.layers.dropout({
                        rate: 0.6
                    }),
                    
                    tf.layers.dense({
                        units: this.numClasses,
                        kernelInitializer: 'varianceScaling',
                        useBias: false,
                        activation: 'softmax'
                    })
                ]
            });
           




            //Compilign model with optimizer 
            this.model.compile({
                optimizer: optimiser, 
                loss: 'categoricalCrossentropy' ,
                metrics: ['accuracy']
             });

             //
             const xs = this.dataset.getXS(); 
             console.log(xs); 
            const batchSize = Math.floor(xs.shape[0] * 0.4);
               if(!(batchSize > 0)) {
                   console.log('Batch size incorrect')
                   return null; 
               }
            
               const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
               const container = {
                 name: 'Model Training', styles: { height: '1000px' }
               };
               const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);
               const earlyStopping = tf.callbacks.earlyStopping({monitor: 'val_loss'}); 
               const callbackList = [fitCallbacks, earlyStopping]

               this.model.fit(
                //Tensor of image features
                this.dataset.getXS(),
                //One-hot encoded labels 
                this.dataset.getYS(),
                 {
                //Option config for batch size, epochs and callback functions. 
                // tf.nextFrame to give browser time to update user interface after each batch 
                    validationSplit: 0.2,
                    batchSize,
                    epochs: trainingEpochs,
                    callbacks: fitCallbacks
               
                }
                
              

                //.then(info => {
                    // console.log('Final accuracy', info.history.val_loss);]
                 //   console.log(JSON.stringify(info, null, 4));              
                //}
                );
                tfvis.show.modelSummary({name: 'Model Architecture'}, this.model);

            


        }

        
                                                             

        checkModel() {
           return this.model.summary(); 

        }

        async saveModel() {
            await this.model.save('downloads://my-model');
            console.log(tf.getBackend())
        }

    }

        


    

       









    

    


 
    