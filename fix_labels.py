from glob import glob

def fix_labels(label_path):
    fixed_lines = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            floats = list(map(float, parts))
            if len(floats) < 5:
                continue

            # Corrigir bbox (primeiros 5 valores)
            cls = int(floats[0])
            x = min(max(floats[1], 0.0), 1.0)
            y = min(max(floats[2], 0.0), 1.0)
            w = min(max(floats[3], 0.0), 1.0)
            h = min(max(floats[4], 0.0), 1.0)

            new_parts = [str(cls), str(x), str(y), str(w), str(h)]

            # Corrigir keypoints (grupos de 3)
            for i in range(5, len(floats), 3):
                kp_cls = int(floats[i])
                kp_x = min(max(floats[i + 1], 0.0), 1.0)
                kp_y = min(max(floats[i + 2], 0.0), 1.0)
                new_parts.extend([str(kp_cls), str(kp_x), str(kp_y)])

            fixed_lines.append(" ".join(new_parts) + "\n")

    with open(label_path, 'w') as f:
        f.writelines(fixed_lines)

# Aplicar a todos os ficheiros de labels
for label_file in glob("train/labels/*.txt") + glob("valid/labels/*.txt"):
    fix_labels(label_file)

print("✔️ Labels corrigidas!")
