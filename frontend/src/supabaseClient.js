import { createClient } from '@supabase/supabase-js';

// Load environment variables. In Vite, env variables must be prefixed with VITE_
const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;

if (!supabaseUrl || !supabaseAnonKey) {
  throw new Error(
    'Missing Supabase environment variables. ' +
    'Please set VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY in frontend/.env'
  );
}

export const supabase = createClient(supabaseUrl, supabaseAnonKey);
