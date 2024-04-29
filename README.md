# COSE-plus

This repository is based on ***COSE***, which is a framework using Bayesian Optimization to successfully found optimal/near-optimal configurations for serverless functions.

COSE-plus aims to find the optimal resource allocation for serverless functions, considering memory and vCPUs respectively.

## Abstract:
소프트웨어의 확장성을 높이고 개발 속도를 높일 수 있다는 점에서 마이크로서비스 아키텍처가 차세대 소프트웨어 개발 방법으로 각광받고 있다. 서버리스 함수는 기존 서버 기반 컴퓨팅 방식과 다르게 인프라를 직접 관리할 필요가 없어 비즈니스 로직에 집중할 수 있는 특징 때문에 마이크로서비스 아키텍처와 통합이 원활하다. 서버리스 함수를 실행하기 위해서는 메모리, vCPU 등 자원을 설정하는 것이 필요하다. 그러나 적절한 자원 설정을 찾는 것은 쉽지 않다. 이를 위해 다양한 방식의 기법이 제안되었으나, 본 연구는 베이지안 최적화 기반 프레임워크 COSE를 개량한 자원 할당 기법을 제시한다. COSE는 비용 및 실행시간 예측을 바탕으로 최적의 메모리 설정 값을 도출하는데, 본 논문에서 제안하는 기법은 메모리와 vCPU를 독립적으로 설정 가능한 변수로 삼아 다양한 워크로드에 유연하게 적용할 수 있다.

## Reference: 
- Nabeel Akhtar, Ali Raza, Vatche Ishakian and Ibrahim Matta<br>
**COSE: Configuring Serverless Functions using Statistical Learning**<br>
*IEEE International Conference on Computer Communications (INFOCOM), 2020* <br>
