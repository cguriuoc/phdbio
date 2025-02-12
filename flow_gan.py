from graphviz import Digraph

def create_gan_flowchart():
    dot = Digraph(format='png')
    dot.attr(rankdir='TB', size='10')
    
    # Nodes
    dot.node('A', 'Load Kidney Dataset')
    dot.node('B', 'Preprocess Images')
    dot.node('C', 'Create DataLoaders')
    dot.node('D', 'Initialize Generator & Discriminator')
    dot.node('E', 'Training Loop (1000 epochs)')
    dot.node('F', 'Train Discriminator')
    dot.node('G', 'Train Generator')
    dot.node('H', 'Save Generated Images')
    dot.node('I', 'Save Final Model')
    
    # Edges
    dot.edge('A', 'B')
    dot.edge('B', 'C')
    dot.edge('C', 'D')
    dot.edge('D', 'E')
    dot.edge('E', 'F', label='For each batch')
    dot.edge('F', 'G', label='After Critic Updates')
    dot.edge('G', 'H', label='Every epoch')
    dot.edge('H', 'E', label='Repeat until 1000 epochs')
    dot.edge('E', 'I', label='End of Training')
    
    dot.render('gan_training_flowchart', view=True)

if __name__ == "__main__":
    create_gan_flowchart()
