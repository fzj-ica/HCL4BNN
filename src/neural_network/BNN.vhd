library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

Package BNN_pack is
    subtype neur is std_logic_vector(2-1 downto 0);
    -- subtype wght is std_logic;
    subtype wght is std_logic_vector(2-1 downto 0);
    subtype bias is std_logic_vector(2-1 downto 0);

    constant neursum_width : integer:= 32;
    subtype neursum is unsigned(neursum_width-1 downto 0);
--    subtype neursum is unsigned;
    
    constant input_width : integer:= 7;
    subtype input_smpl is unsigned(input_width-1 downto 0);
    
    type neursum_map is array (0 to 3-1) of integer;

    function neur_int ( a: neur) return neursum;
    function neur_int ( a: integer ) return neursum;
    function neur_ws ( a: std_logic; w: wght) return neursum;
    function neur_ws ( a: neur; w: wght) return neursum;
    function neur_act ( a: neursum; nmap: neursum_map) return neur;

    function inp_ws ( a: input_smpl; w: wght) return neursum;


--    function "+" ( L: neur; R: neur) return neur;
--    function "*" ( L: neur; R: wght) return neursum;
--    function "*" ( L: wght; R: neur) return neursum;
--    function "+" ( L: neur; R: neur) return neursum;
--    function "+" ( L: neursum; R: neur) return neursum;
--    function "+" ( L: neursum; R: neursum) return neursum;
--    function "+" ( L: neur; R: neursum) return neursum;
end;

Package body BNN_pack is


--    function "+" ( L: neur; R: neur) return neursum is begin
--      return (neur_int(L) + neur_int(R));
--    end;

--    function "+" ( L: neursum; R: neur) return neursum is begin
--      return (L + neur_int(R));
--    end;

--    function "+" ( L: neur; R: neursum) return neursum is begin
--      return (neur_int(L) + R);
--    end;

--    function "+" ( L: neursum; R: neursum) return neursum is begin
--      return (L + R);
--    end;

--    function "*" ( L: neur; R: wght) return neursum is begin
--      return neur_ws(L,R);
--    end;
--    function "*" ( L: wght; R: neur) return neursum is begin
--      return neur_ws(R,L);
--    end;
    
    

    function neur_act ( a: neursum; nmap: neursum_map) return neur
    is 
--      variable res: unsigned;
    begin
--      if    ( a >=                                 0  and a < to_unsigned(nmap(0),neursum_width) ) then return b"00";
      if    ( a >= to_unsigned(nmap(0),neursum_width) and a < to_unsigned(nmap(1),neursum_width) ) then return b"01";
      elsif ( a >= to_unsigned(nmap(1),neursum_width) and a < to_unsigned(nmap(2),neursum_width) ) then return b"10";
      elsif ( a >= to_unsigned(nmap(2),neursum_width)  ) then return b"11";
      else return b"00";
      end if;
    end;



    function neur_int ( a: neur ) return neursum
    is 
      variable res: neursum;
    begin
        res := resize(unsigned( a ) , neursum_width);
        return res; 
    end;


    function neur_int ( a: integer ) return neursum
    is 
      variable res: neursum;
    begin

        res := to_unsigned( a, neursum_width );
--        res := resize(unsigned( a ) , neursum_width);
        return res; 
    end;


    function neur_ws ( a: neur; w: wght) return neursum
    is 
        variable sel : std_logic_vector(3 downto 0);
        variable res: neur;
    begin
        sel := w & a;
        case sel is
          when "0000" => res := b"00"; -- w="00", a="00"
          when "0001" => res := b"00"; -- w="00", a="01"
          when "0010" => res := b"00"; -- w="00", a="10"
          when "0011" => res := b"00"; -- w="00", a="11"
          when "0100" => res := b"00"; -- w="01", a="00"
          when "0101" => res := b"01"; -- w="01", a="01"
          when "0110" => res := b"10"; -- w="01", a="10"
          when "0111" => res := b"11"; -- w="01", a="11"
          when "1000" => res := b"01"; -- w="10", a="00"
          when "1001" => res := b"10"; -- w="10", a="01"
          when "1010" => res := b"11"; -- w="10", a="10"
          when "1011" => res := b"11"; -- w="10", a="11"
          when "1100" => res := b"11"; -- w="11", a="00"
          when "1101" => res := b"10"; -- w="11", a="01"
          when "1110" => res := b"01"; -- w="11", a="10"
          when "1111" => res := b"00"; -- w="11", a="11"
          when others => res := b"01";
        end case;
        return resize(unsigned( res ) , neursum_width); 
    end;

    function neur_ws ( a: std_logic; w: wght) return neursum
    is 
        variable sel : std_logic_vector(2 downto 0);
        variable res: neur;
    begin
        sel := w & a; -- a=0 => 1; a=1 => 2
        case sel is
          when "000" => res := b"00"; -- w="00", a="1"
          when "001" => res := b"00"; -- w="00", a="2"
          when "010" => res := b"01"; -- w="01", a="1"
          when "011" => res := b"10"; -- w="01", a="2"
          when "100" => res := b"10"; -- w="10", a="1"
          when "101" => res := b"11"; -- w="10", a="2"
          when "110" => res := b"11"; -- w="11", a="1"
          when "111" => res := b"00"; -- w="11", a="2"
          when others => res := b"01";
        end case;
        return resize(unsigned( res ) , neursum_width); 
    end;


    function inp_ws ( a: input_smpl; w: wght) return neursum
    is 
        variable sel : std_logic_vector(w'LENGTH-1 downto 0);
        variable val, res: neursum;
        constant maxval: neursum := to_unsigned((2**input_width)-1,neursum_width);
    begin
        sel := w ; -- a = 5 bit
        val := resize( a , neursum_width);
        case sel is
          when "00" => res := (others => '0');                -- w="00", a --> 0
          when "01" => res := maxval and (not val);  -- w="01", a --> a >> 1
          when "10" => res := val;                -- w="10", a --> a
          when "11" => res := shift_left(val,1) when a(a'LENGTH-1)='0' else maxval;-- shift_right(val,1); -- w="11", a --> a << 1
          when others => res := (others => '0');
        end case;
        return resize( res , neursum_width); 
    end;

    

    
end;