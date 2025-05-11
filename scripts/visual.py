import matplotlib.pyplot as plt

def plot_fitness(fitness_list):
    # 세대 수를 x축 값으로 생성
    generations = list(range(1, len(fitness_list) + 1))

    # 그래프 그리기
    plt.figure(figsize=(10, 6))
    plt.plot(generations, fitness_list, marker='o', linestyle='-', color='b', label='Fitness Score')

    # 그래프 제목과 축 레이블 설정
    plt.title('Generation vs length')
    plt.xlabel('Generation')
    plt.ylabel('length')

    # 범례 추가
    plt.legend()

    # 그리드 추가
    plt.grid(True)

    # 그래프 보여주기
    plt.show()