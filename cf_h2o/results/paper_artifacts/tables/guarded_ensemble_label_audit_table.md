| component                 | uses_source_labels | uses_unlabeled_target | uses_target_transition_labels | audit_status                                           |
| ------------------------- | ------------------ | --------------------- | ----------------------------- | ------------------------------------------------------ |
| source residual fitting   | yes                | no                    | no                            | source-city supervised residual only                   |
| target similarity weights | no                 | yes                   | no                            | obs/action, simulator output, static local descriptors |
| feature-set selector      | yes                | no                    | no                            | leave-one-source score before target evaluation        |
| leave-one-source selector | yes                | yes                   | no                            | replay matched                                         |
| ensemble dominance guard  | no                 | yes                   | no                            | max source weight threshold only                       |
| reported target metrics   | no                 | no                    | yes                           | evaluation only; not fed back to selector              |
