# 🚀 Launch the Topic Classification App

Bring your topic classification demo to life — whether you're a Docker enthusiast or prefer a lightweight local setup, we’ve got you covered.

---

## 🐳 Run It Locally with Docker (Fast & Isolated)

1. **Choose Your Dataset Flavor**
   Navigate to either of these directories depending on the dataset you want to explore:

   * `Docker_deployment/demo_20NG` (20 Newsgroups)
   * `Docker_deployment/demo_r8` (Reuters-8)

2. **Build the Docker Image**
   Run the following command to build a fresh image (no cache used):

   ```bash
   docker build --no-cache -t topic-classification-app .
   ```

3. **Start the Container**
   Fire up the app by mapping port 7860:

   ```bash
   docker run -p 7860:7860 topic-classification-app
   ```

4. **Launch in Your Browser**
   Open [http://localhost:7860](http://localhost:7860) and enjoy the interactive demo!

---

## 🧪 Prefer a Pure Python Setup? No Problem!

1. **Navigate to the Demo Folder**
   Choose the dataset folder you want to explore:

   * `Docker_deployment/demo_20NG`
   * `Docker_deployment/demo_r8`

2. **Install Python Dependencies**
   Set up your environment:

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure the App Launch Option**
   Open `app.py` and locate the `app.launch()` line.

   Choose one of the following based on your needs:

   * **Local access only**:

     ```python
     app.launch()
     ```

     Open [http://localhost:7860](http://localhost:7860) to use the app locally.

   * **Want to share your app online?**
     Enable public sharing with Gradio:

     ```python
     app.launch(share=True)
     ```

     A shareable public URL will be generated for you.

---

✨ That's it! You're now ready to explore topic classification models with ease and flair. Happy experimenting!
