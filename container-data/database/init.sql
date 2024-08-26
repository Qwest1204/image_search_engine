create table public.chroma_images (
  uniq_id character varying not null,
  np_array_bytes bytea,
  image_bytes bytea,
  some_text text
);

