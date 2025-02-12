from graphviz import Digraph

def create_flowchart(output_path='flowchart.png'):
    dot = Digraph(format='png')
    
    # Nodes
    dot.node('A', 'Start Training')
    dot.node('B', 'Loop Over Epochs')
    dot.node('C', 'Loop Over Training Data')
    dot.node('D', 'Train Step (Generator & Discriminator)')
    dot.node('E', 'Every 5 Epochs: Generate & Save Images')
    dot.node('F', 'Save Model Checkpoints')
    dot.node('G', 'Training Complete')
    
    # Edges
    dot.edge('A', 'B')
    dot.edge('B', 'C', label='For each epoch')
    dot.edge('C', 'D', label='For each batch')
    dot.edge('D', 'C', label='Repeat for all batches')
    dot.edge('C', 'E', label='Every 5 epochs')
    dot.edge('E', 'B', label='Continue training')
    dot.edge('B', 'F', label='After all epochs')
    dot.edge('F', 'G')
    
    # Render and save the flowchart
    dot.render(output_path, format='png', cleanup=True)
    print(f'Flowchart saved as {output_path}.png')

if __name__ == "__main__":
    create_flowchart()

