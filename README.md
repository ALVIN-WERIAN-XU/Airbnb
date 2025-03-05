# Airbnb

Abstract

Airbnb is a platform that offers accommodation rental services and has gradually
become one of the mainstream options for travel lodging worldwide. As Airbnb
continues to gain popularity, this study aims to recommend properties from its
database based on user preferences. This project involves analyzing 27 distinct
attributes of rooms and integrating images from 74,112 different properties. It
uses deep convolutional neural networks (CNNs), fine-tunes the ResNet50 model,
and utilizes natural language processing (NLP) techniques.
The study shows that specific property details, such as furniture, room type,
bed type, and number of rooms, play a crucial role in image recognition. By
identifying, deconstructing, and analyzing user-input keywords or images, the
system can infer user preferences and recommend properties from the database
that best match their needs. Finally, the performance of the recommendation
system is evaluated by generating synthetic labels and calculating metrics such as
accuracy and loss.

1.Introduction

With the rise of the sharing economy, Airbnb, as one of the largest accommodation
sharing platforms, has become a preferred choice for many travelers. When selecting
Airbnb properties, users often seek accommodations that align with their specific
needs and preferences, influenced by factors such as property type, room layout, and
available amenities. However, given the vast number of listings on the platform,
finding the right property can be a time-consuming task for users. Therefore, designing
an intelligent recommendation system that can quickly filter and identify key listing
words based on user preferences becomes particularly significant.
This project starts by selecting a subset of features from 27 available attributes,
including property type, number of bedrooms, number of beds, and amenities. Using
a pre-trained ResNet50 model as a feature extractor, an efficient mechanism is built
to extract deep features from property images while standardized numerical features,
multi-label encoded amenities, and one-hot encoded categorical features are combined
to create unified feature vectors. By calculating the cosine similarity between the
feature vectors of user-uploaded images and the property features in the database, the
system identifies the most similar listings and generates a ranked recommendation list
based on similarity scores. To further enhance the model’s generalization capability,
data augmentation techniques are applied to the training images in real-time, improving
the model’s adaptability to diverse data. Ultimately, the system provides personalized
property recommendations to users based on uploaded images and specific input
criteria (e.g., property type, amenities), enhancing the overall user experience.

2.Background

2.1 Industry Background of Airbnb

Since its establishment in 2008, Airbnb has been dedicated to creating an online
platform that connects short-term rental hosts with guests. Over the years, it has
rapidly grown into a global leader in the shared accommodation industry. Compared
to traditional hotels, Airbnb offers greater flexibility and diversity, allowing it to
better meet users’ personalized needs. Moreover, during their stay, users can gain
a deeper appreciation of local culture and traditions. However, with the continuous
growth of the platform’s user base and the number of listings, the challenge of quickly
and accurately identifying accommodations that align with users’ preferences has
become increasingly prominent. As a result, designing intelligent and personalized
recommendation systems based on user needs has emerged as a key focus of research.

2.2 Challenges in Recommendation Systems

Existing recommendation systems primarily rely on user behavior data and explicit
preferences. However, this not only violates user privacy, but the multimodal data
of listings adds complexity to the design of recommender systems. Among these
modalities, image data plays an irreplaceable role in showcasing property features, yet
traditional recommendation systems often struggle to effectively extract and utilize
such visual information. Moreover, integrating multimodal data while improving the
accuracy and relevance of recommendations remains a critical challenge.

2.3 Applications of Deep Learning in Recommendation Systems

In recent years, deep convolutional neural networks (CNNs) and natural language
processing (NLP) have demonstrated immense potential across various fields. CNNs
excel at extracting image features, capturing high-dimensional characteristics, while
NLP techniques effectively quantify the sentiment and semantics of textual descriptions
(Arif 2020). Both approaches are well-suited to addressing the challenges of Airbnb’s
recommendation mechanisms. However, despite their theoretical compatibility with
Airbnb’s data, practical implementation still faces significant challenges, such as
computational complexity and model interpretability.

2.4 Innovation

To address the challenges mentioned above, this study adopts a multimodal recommendation system framework that integrates deep learning techniques. 
By extracting image and textual features and combining them with other property attributes, the system
aims to deliver personalized recommendations to users. Leveraging the proposed
models and techniques, the study seeks to explore how to efficiently utilize multimodal
data to enhance the performance of recommendation systems.

3.Data

This dataset includes information such as property type, property description, room
type, amenities, bedroom configuration, city location, and room thumbnail links
(“thumbnail_URL”). During the data preprocessing stage, initial data cleaning was
performed, and some variables were further refined for improved usability.

3.1 Textual Data Quantification with NLP

To simplify the representation of property descriptions in the dataset, this project
introduced a new variable called “sentiment_score”. In the original dataset, property
descriptions were presented in textual form, which could not be directly used for
quantitative analysis. Therefore, a function named “score_description” was designed,
leveraging natural language processing (NLP) techniques to evaluate the descriptions
across multiple dimensions. By calculating weighted scores for keyword density,
grammar, sentiment, and fluency, the function generates an overall score that reflects
the overall quality of the descriptions. In subsequent analysis, the “sentiment_score” is
used to represent the descriptions of different properties. Meanwhile, the ‘description‘
variable is removed at this stage.

3.2 Handling Missing Values

It is noticeable that the original data contains some missing values, specifically listings
that did not provide room images (i.e., “thumbnail URL” was missing). Since the goal
of this study is to extract information and perform analysis based on images, listings
without images could not be effectively included in the analysis. Consequently, these
records with missing images were removed from the dataset.

3.3 Remove Unused Variables

The dataset contains complex variables, so some irrelevant variables will be removed
during the data preprocessing stage. Since the core of this recommendation system is
to match properties based on keywords and images provided by users, certain variables
(such as the latitude, longitude, or postal code of a property) are unlikely to be used as
filtering criteria by users and will therefore be excluded.

3.4 Split Dataset

Since the recommendation mechanism in this study aims to identify properties that best
match user preferences from the database, part of the data needs to be used for model
training, while the remaining data serves as the database for the recommendation
system. To prevent data leakage, the data used for training the model should not
be reused for constructing the database used in recommendations. To achieve this,
the original dataset was randomly divided into two parts: one-third of the data was
allocated for model research and training, while the remaining two-thirds were used
as a database encompassing all available Airbnb listings for matching and retrieval
within the recommendation system. As a result, the original dataset was split into
two separate files, “train_data” and “database_data”, to be used in different stages for
model training and database construction, respectively.

3.5 Data Augmentation

To enhance the robustness and generalization capability of the model, this study applied
various data augmentation techniques to the training data. These included horizontal
flipping and random rotation of images, designed to simulate potential directional
variations in property images and improve the model’s adaptability to images captured
from different angles. Additionally, pixel normalization was employed, scaling image
pixel values to the range [0, 1], which helped to smooth the input data distribution.
Furthermore, the data was combined with image features by incorporating onehot encoding, standardization, and multi-label binarization for the tabular features,
resulting in a comprehensive feature representation.

4 Methodology

Given the high complexity of images in the dataset and the significant variation in their
features, this project explores various image feature recognition techniques to address
these challenges. Based on the characteristics of the dataset, a custom convolutional
neural network (CNN), the ResNet50 model, and a fine-tuned version of ResNet50
were employed for image recognition. Principal Component Analysis To address the
high-dimensional nature of the dataset and identify the most important features, this
study employed Principal Component Analysis (PCA) for dimensionality reduction
while evaluating feature importance. Using the PCA component weight matrix and
explained variance ratios, the contribution of each original feature to the principal
components was calculated. Subsequently, the contributions were weighted by the
importance of the principal components and summed to derive the final importance
score for each feature.

4.1 Custom Convolutional Neural Network

To further explore the data in this dataset, the project initially employed a custom
convolutional neural network (CNN) as a preliminary attempt. However, due to its
simplistic design—comprising only three convolutional layers and one fully connected
layer—the model’s depth was limited, and its parameters were relatively few (Alzubaidi
2021). Consequently, it could only capture basic low-level features, such as edges and
textures, while lacking the ability to effectively represent complex scenes. As a result,
the custom CNN proved inadequate for robust image feature extraction.

4.2 ResNet50 Model

ResNet50 is capable of learning features at deeper levels while maintaining training
stability, making it highly effective for extracting higher-order features. Additionally,
its relatively low computational cost makes it well-suited for the needs of this study
(Vogt 2019). In scenarios with limited image data, ResNet50’s transfer learning
capabilities effectively mitigate the issue of insufficient training data. Overall, selecting
ResNet50 as the base model strikes an optimal balance between performance and
computational cost. It not only ensures the stability of deep network training but
also provides robust general feature extraction capabilities. However, given the
characteristics of this dataset, the ResNet50 model still faces limitations in certain
aspects. While the output images often exhibit high similarity scores with the input
images, they are not truly similar in terms of meaningful features.

4.3 Fine-Tuning Model from ResNet50 Model

To address the specific requirements of this study and overcome the limitations of
the ResNet50 model in handling the task, this research fine-tuned ResNet50 to better
adapt to the extraction of property image features. The fine-tuned model removes the
fully connected layers of ResNet50 and incorporates a global average pooling layer to
extract fixed-length deep feature vectors. Additionally, the first 140 layers of ResNet50
were frozen to retain general features and focus on optimizing high-level features.
This design reduces computational costs and minimizes the risk of overfitting. A
custom top layer was designed, including a fully connected layer to learn high-order
features from property images, a dropout layer to enhance the model’s generalization
capability, and an output layer for the binary classification task. The training objective
of the fine-tuned model is to minimize the binary cross-entropy loss and optimize the
model’s performance.

4.4 Recommendation System Design

To achieve property recommendations based on user inputs, this study designed a
recommendation system that combines property images and tabular features to identify
the most relevant listings through similarity calculations. The system first preprocesses
user-uploaded images by resizing them to a fixed size and extracting deep features.
If the input includes tabular features such as property type, number of bedrooms,
and amenities, the system encodes and standardizes these features. If no tabular
features are provided, the system uses default values (e.g., property type: "House,"
number of bedrooms: 3, amenities: "TV" and "Internet"). Finally, the image features
and tabular features are merged into a comprehensive feature vector for matching
properties in the database. The system evaluates the similarity between user inputs
and each property in the database using cosine similarity. To enhance the relevance of
the recommendations, the system prioritizes properties located in the same city as
specified by the user and sorts the properties based on similarity scores. Ultimately,
the system returns the top 5 most similar properties, including details such as property
ID, type, image, and more.

5. Analysis and Discussion

This section explains how the aforementioned methods were used to analyze the
data and provides a detailed overview of the program’s operational logic. Since the
custom CNN model was only utilized for exploratory purposes and its performance
was unsatisfactory, it will not be discussed in detail.
Additionally, due to the large size of the dataset, downloading images is timeconsuming, and running the program and training the model 
incur significant computational costs. Therefore, this study randomly selected and downloaded 10,000
images from the ‘train_data’ dataset, which were then split into two parts in a 2:1
ratio: one for model construction and the other for building the property database.
After completing the model setup and tuning, the model was further applied to the
full dataset (full train_dataset) for additional debugging and refinement.

5.1 Dimensionality Reduction and Feature Importance Visualization

The dataset is composed of the following features:
• Image features: Extracted using the pre-trained ResNet50 model, resulting in
2048-dimensional feature vectors.
• Categorical features: Encoded with OneHotEncoder, contributing 15 dimensions.
• Numerical features: Standardized, contributing 2 dimensions.
• Multi-label features: Encoded with MultiLabelBinarizer, generating 194 dimensions.
These features were combined to form a complete feature matrix with 2259 columns.
To reduce computational complexity while retaining key information, PCA was
applied to the feature matrix, preserving the top 10 principal components. These
10 components explain the majority of the variance in the dataset. Analysis results
indicate that the following features contribute the most to explaining the variance:
the number of beds, number of bedrooms, ‘property_type’, and the availability of
internet among the amenities, with importance scores of 0.08, 0.066, 0.056, and 0.038,
respectively.
Next, a further selection and combination were performed among the top 10 most
influential features. Ultimately, "house type," "internet availability in facilities,"
"number of rooms," and "number of beds" were chosen as the feature set. This
combination retains the original information to the greatest extent while reducing the
impact of noise on model performance during fitting. Hence, these variables will be
used as input features (‘x’) in the subsequent model tuning process

5.2 Fine Tuning Model

In this project, the pre-trained ResNet50 was selected as the base model, with its
weights already trained on the ImageNet dataset. This approach significantly reduces
training time while enhancing the model’s ability to capture general image features
effectively.
The initial experimental results were less than satisfactory, with low prediction
accuracy and cosine similarity, leading to suboptimal recommendation quality. The
model struggled to effectively identify features within the input images. Below are
examples of input and output images(Figure 2-5), illustrating that only a subset
of image features were accurately recognized. This issue can be attributed to the
imbalance in feature scales, where features with larger magnitudes dominated the
training process, overshadowing others.

From the above examples, it can be observed that certain features in the input
images (such as the sofa) were effectively recognized in some cases. Some output
images appear to lack a clear connection with the input images.
However, the image similarity is unusually high (Top 5 Similarity distribution:
[0.96960488 0.9819001 0.99564546 0.9980545 0.99891162]), which is almost
impossible given the dataset used in this study, as the data volume is insufficient to
support such high similarity. Additionally, the model exhibits extremely high accuracy
and very low loss on the validation set (Table 2), further indicating the potential
occurrence of overfitting.

5.3 Normalization

To enhance the model’s performance, numerical features were normalized. Normalization allows the model to better leverage information from different features, thereby
improving the accuracy of classification and recommendations. This approach is
particularly effective for the multi-modal dataset used in this study, which combines
image and tabular data, as the feature value ranges across different data types often
vary significantly.
Below are the model output examples after normalization, given the same input. From
these examples, it can be observed that although the similarity has slightly decreased
(Top 5 Normalized Similarity distribution: [0.40481358 0.50266529 0.54152543
0.57094769 0.69440439] ), the model’s ability to recognize various features has
significantly improved. The input image depicts a room with two sofas and a table,
and the output images have successfully captured these key features, demonstrating an
enhancement in the model’s performance. Additionally, the model’s accuracy and the
loss also has been a marked improvement (Table 10).

6. Conclusion

This study used convolutional neural networks (CNNs) and other techniques to
deeply analyze the features of Airbnb property images and develop a property
recommendation mechanism. Through initial data cleaning, PCA-based dimensionality reduction, and feature selection, the complexity of the data was effectively
reduced, significantly improving the accuracy of subsequent models. Furthermore, by
comprehensively evaluating the model’s accuracy, loss, and output image similarity,
this study successfully developed a property recommendation system capable of
effectively identifying user needs, providing reliable technical support for personalized
recommendations.

Overall, the model without normalization exhibited clear overfitting issues, while
the fine-tuned model with normalization demonstrated stronger generalization
capabilities, despite slightly lower similarity and accuracy. This indicates that the
model can more effectively analyze features when applied to unseen datasets. In fact,
the output image results further confirm this observation—normalized outputs are
more closely aligned with the features of the input images, showcasing the model’s
improved ability to accurately identify key features.
However, there is still room for improvement in this project. After data cleaning, the
dataset contained 74,112 property details. To reserve part of the data for the database,
the dataset was split into a 1/3 : 2/3 ratio, with 24,704 entries used for model training
and 49,408 entries allocated for the database. However, the dataset size remains
relatively small for image analysis tasks. If all 74,112 entries were used for model
training and additional data (e.g., 200,000 more property records) were introduced as
the database, the model’s performance and capability could be significantly enhanced,
providing stronger support for the accuracy and reliability of the recommendation
system.This project has provided us with valuable insights into deep learning techniques and their application to real-world problems. We would like to express our
heartfelt gratitude to Prof. Kathryn and the TAs for their guidance and support
throughout the project and the paper.

References

Arif Wani, M., Kantardzic, M., & Sayed-Mouchaweh, M. (2020). Trends in deep
learning applications. Deep learning applications, 1-7. [Online].
Available: https://link.springer.com/chapter/10.1007/978-981-15-1816-4_1
Alzubaidi, L., Zhang, J., Humaidi, A. J., Al-Dujaili, A., Duan, Y., Al-Shamma, O., &
Farhan, L. (2021). Review of deep learning: concepts, CNN architectures, challenges,
applications, future directions. Journal of big Data, 8, 1-74.[Online].
Available: https://link.springer.com/article/10.1186/s40537-021-00444-8
James, G., Witten, D., Hastie, T., Tibshirani, R., & Taylor, J. (2023). An introduction
to statistical learning: With applications in Python. Springer Nature.[Online].
Available: https://www- springer-com.proxy.lib.uwaterloo.ca/series/0417
Vogt, M. (2019). An overview of deep learning and its applications. Fahrerassistenzsysteme 2018: Von der Assistenz zum automatisierten Fahren 4. Internationale
ATZ-Fachtagung Automatisiertes Fahren, 178-202.[Online].
Available:https://link.springer.com/chapter/10.1007/978-3-658-23751-6_17
