import type { ExampleItem } from "../types";

type ExampleGridProps = {
  items: ExampleItem[];
};

export default function ExampleGrid({ items }: ExampleGridProps) {
  if (items.length === 0) {
    return <p className="loading-copy">Loading examples...</p>;
  }

  return (
    <div className="example-grid">
      {items.map((item) => (
        <article key={item.id} className="example-card">
          <img src={item.image} alt={`${item.category} example`} className="example-image" />
          <div className="example-meta">
            <p className="example-title">
              {item.category} · {item.status}
            </p>
            <p className="example-id">{item.id}</p>
          </div>
        </article>
      ))}
    </div>
  );
}
