import circuit
import gate
import ring


def main():
    # Parameters
    n = 20
    chi = 64

    # Generate circuit
    hadamard_layer = circuit.Circuit(n)
    for i in range(n):
        hadamard_layer.add_gate(i, gate.H())

    # Simulation
    sim = ring.Ring(n, chi)
    sim.run(hadamard_layer)


if __name__ == '__main__':
    main()
