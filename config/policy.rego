# Ayojna placement policy for Open Policy Agent (OPA 1.x, Rego v1).
#
# The RULES live here; their PARAMETERS come from config/policy.yaml (sent as input.policy),
# so OPA and the built-in Python guard read one source of truth and are cross-checked.
#
# Query:  POST /v1/data/ayojna/decision  {"input": {"policy": {...}, "volumes": [...]}}
# Answer: per volume -> allowed tiers, freeze (legal hold) and the reasons.
package ayojna

tiers := ["hot", "warm", "cold", "archive"]

# index of the slowest tier this SLA class may use (unknown class -> archive)
floor_index(sla) := i if {
	some i, t in tiers
	t == object.get(input.policy.sla_floor, sla, "archive")
}

archive_blocked(v) if v.data_class in input.policy.no_archive_classes

frozen(v) if {
	v.legal_hold == true
	input.policy.legal_hold_freezes == true
}

allowed(v) := [t |
	some i, t in tiers
	i <= floor_index(v.sla_class)
	not blocked(v, t)
]

blocked(v, "archive") if archive_blocked(v)

floor_reason(v) := [sprintf("%s SLA: not below %s", [v.sla_class, tiers[floor_index(v.sla_class)]])] if {
	floor_index(v.sla_class) < 3
} else := []

archive_reason(v) := [sprintf("%s: no archive", [v.data_class])] if archive_blocked(v) else := []

hold_reason(v) := ["legal hold: frozen"] if frozen(v) else := []

decision[v.name] := {
	"allowed": allowed(v),
	"freeze": is_frozen_value(v),
	"reasons": array.concat(array.concat(floor_reason(v), archive_reason(v)), hold_reason(v)),
} if {
	some v in input.volumes
}

is_frozen_value(v) := true if frozen(v) else := false