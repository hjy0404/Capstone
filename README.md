model의 output은 [a,b,c,d,e] 이며 각각은 label 별 확률을 의미함.따라서 torch.max 를 사용하여 최대 확률의 label을 추출해내야함.
