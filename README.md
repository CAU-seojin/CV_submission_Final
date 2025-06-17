# CV_submission_Final

### submission_20220515.py: LCNet
---
1. 가상환경 생성

`conda create -n cv pthon=3.10`

2. 가상환경 활성화
 
`conda activate cv`

3. Jupyter Kernel 등록
   
`pip install ipykernel`

`python -m ipykernel install --user --name cv --display-name cv`

4. 필요 패키지 설치
    
`pip install -r requirements.txt`

---
`requirements.txt` 설치할 때, 꼭 cu118을 확인할 것! 이는 device와 연계됨!!
